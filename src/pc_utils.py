# coding=utf-8
import tensorflow as tf
import scipy.sparse
from sklearn.neighbors import KDTree
import numpy as np
import math
import multiprocessing as multiproc
from functools import partial

def GridSampling(batch_size, meshgrid):
    '''
    output Grid points as a NxD matrix

    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    '''

    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD 2025x2
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
    return g


def jitter_perturbation_point_cloud(batch_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    assert (clip > 0)
    jittered_data = tf.clip_by_value(sigma * tf.random_normal(tf.shape(batch_data)), -1 * clip, clip)
    jittered_data = tf.concat([batch_data[:, :, :3] + jittered_data[:, :, :3], batch_data[:, :, 3:]], axis=-1)
    return jittered_data


def rotate_point_cloud_and_gt(batch_data, batch_gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    # batch_data = tf.expand_dims(batch_data,axis=0)
    batch_size, num_point, num_channels = batch_data.get_shape().as_list()

    angles = tf.random_uniform((batch_size, 3), dtype=tf.float32) * 2 * np.pi
    cos_x, cos_y, cos_z = tf.split(tf.cos(angles), 3, axis=-1)  # 3*[B, 1]
    sin_x, sin_y, sin_z = tf.split(tf.sin(angles), 3, axis=-1)  # 3*[B, 1]
    one = tf.ones_like(cos_x, dtype=tf.float32)
    zero = tf.zeros_like(cos_x, dtype=tf.float32)
    # [B, 3, 3]
    Rx = tf.stack(
        [tf.concat([one, zero, zero], axis=1),
         tf.concat([zero, cos_x, sin_x], axis=1),
         tf.concat([zero, -sin_x, cos_x], axis=1)], axis=1)

    Ry = tf.stack(
        [tf.concat([cos_y, zero, -sin_y], axis=1),
         tf.concat([zero, one, zero], axis=1),
         tf.concat([sin_y, zero, cos_y], axis=1)], axis=1)

    Rz = tf.stack(
        [tf.concat([cos_z, sin_z, zero], axis=1),
         tf.concat([-sin_z, cos_z, zero], axis=1),
         tf.concat([zero, zero, one], axis=1)], axis=1)

    rotation_matrix = tf.matmul(Rz, tf.matmul(Ry, Rx))

    if num_channels > 3:
        batch_data = tf.concat(
            [tf.matmul(batch_data[:, :, :3], rotation_matrix),
             tf.matmul(batch_data[:, :, 3:], rotation_matrix),
             batch_data[:, :, 6:]], axis=-1)
    else:
        batch_data = tf.matmul(batch_data, rotation_matrix)

    if batch_gt is not None:
        if num_channels > 3:
            batch_gt = tf.concat(
                [tf.matmul(batch_gt[:, :, :3], rotation_matrix),
                 tf.matmul(batch_gt[:, :, 3:], rotation_matrix),
                 batch_gt[:, :, 6:]], axis=-1)
        else:
            batch_gt = tf.matmul(batch_gt, rotation_matrix)

    return batch_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    batch_size, num_point, num_channels = batch_data.get_shape().as_list()
    angles = tf.clip_by_value(tf.random_normal((batch_size, 3)) * angle_sigma, -angle_clip, angle_clip)

    cos_x, cos_y, cos_z = tf.split(tf.cos(angles), 3, axis=-1)  # 3*[B, 1]
    sin_x, sin_y, sin_z = tf.split(tf.sin(angles), 3, axis=-1)  # 3*[B, 1]
    one = tf.ones_like(cos_x, dtype=tf.float32)
    zero = tf.zeros_like(cos_x, dtype=tf.float32)
    # [B, 3, 3]
    Rx = tf.stack(
        [tf.concat([one, zero, zero], axis=1),
         tf.concat([zero, cos_x, sin_x], axis=1),
         tf.concat([zero, -sin_x, cos_x], axis=1)], axis=1)

    Ry = tf.stack(
        [tf.concat([cos_y, zero, -sin_y], axis=1),
         tf.concat([zero, one, zero], axis=1),
         tf.concat([sin_y, zero, cos_y], axis=1)], axis=1)

    Rz = tf.stack(
        [tf.concat([cos_z, sin_z, zero], axis=1),
         tf.concat([-sin_z, cos_z, zero], axis=1),
         tf.concat([zero, zero, one], axis=1)], axis=1)


    rotation_matrix = tf.matmul(Rz, tf.matmul(Ry, Rx))

    if num_channels > 3:
        batch_data = tf.concat(
            [tf.matmul(batch_data[:, :, :3], rotation_matrix),
             tf.matmul(batch_data[:, :, 3:], rotation_matrix),
             batch_data[:, :, 6:]], axis=-1)
    else:
        batch_data = tf.matmul(batch_data, rotation_matrix)

    return batch_data


def random_scale_point_cloud_and_gt(batch_data, batch_gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.get_shape().as_list()
    scales = tf.random_uniform((B, 1, 1), minval=scale_low, maxval=scale_high, dtype=tf.float32)

    batch_data = tf.concat([batch_data[:, :, :3] * scales, batch_data[:, :, 3:]], axis=-1)

    if batch_gt is not None:
        batch_gt = tf.concat([batch_gt[:, :, :3] * scales, batch_gt[:, :, 3:]], axis=-1)

    return batch_data



if __name__ == '__main__':
    batch_size = 8
    meshgrid = [[0, 1, 16], [0, 1, 16], [0, 1, 16]]
    grid = GridSampling(batch_size, meshgrid)