import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
interpolate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so'))
def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn(xyz1, xyz2)
ops.NoGradient('ThreeNN')
def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.three_interpolate(points, idx, weight)
@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]

def pointnet_fp_module(xyz1, xyz2, points, scope):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points: (batch_size, ndataset1, nchannel1) TF tensor
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points, idx, weight)
        new_point = tf.concat(axis=1, values=[interpolated_points, points]) # B,ndataset1,nchannel
        print(new_point.shape)
        return new_point



if __name__=='__main__':
    import numpy as np
    import time
    # np.random.seed(100)
    # tf.enable_eager_execution()
    #
    # f = open('test.pts', 'r+')
    # flist = f.readlines()
    # lines = [line.split( ) for line in flist[:]]
    # slice = random.sample(lines, 256)
    # array =np.array(slice)
    #
    # array = np.expand_dims(array, axis=0)
    # with tf.device('/cpu:0'):
    #     pts = array.astype('float32')
    #     xyz1 = array.astype('float32')
    #     xyz2 = array.astype('float32')
    #     points = tf.constant(pts)
    #     xyz1 = tf.constant(xyz1)
    #     xyz2 = tf.constant(xyz2)
    #     for i in range(2):
    #         # dist, idx = three_nn(xyz1, xyz2)
    #         # weight = tf.ones_like(dist)/3.0
    #         # interpolated_points = three_interpolate(points, idx, weight)
    #         # new_point = tf.concat([interpolated_points,points], axis=1)
    #         new_point = pointnet_fp_module(pts, pts, pts, scope='expand_1')
    #         points = new_point
    #         xyz1 = new_point
    #         xyz2 = new_point
    #     # with tf.Session('') as sess:
    #     #     now = time.time()
    #     #     for _ in range(100):
    #     #         ret = sess.run(interpolated_points)
    #     #     print(time.time() - now)
    #     #     print(ret.shape, ret.dtype)
    # print(new_point.shape)
    # x_np = tf.squeeze(new_point, axis=0)
    # # xyz2 = tf.squeeze(xyz2, axis=0)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # array = tf.squeeze(array, axis=0)
    # print(len(x_np))
    # ax.scatter(x_np[:, 0], x_np[:, 1], x_np[:, 2], s=1, color='red')
    # # ax.scatter(array[:, 0], array[:, 1], array[:, 2], s=1, color='green')
    # plt.show()

    np.random.seed(100)
    pts = np.random.random((32, 128, 64)).astype('float32')
    tmp1 = np.random.random((32, 512, 3)).astype('float32')
    tmp2 = np.random.random((32, 128, 3)).astype('float32')
    with tf.device('/cpu:0'):
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        dist, idx = three_nn(xyz1, xyz2)
        weight = tf.ones_like(dist) / 3.0
        interpolated_points = three_interpolate(points, idx, weight)
    with tf.Session('') as sess:
        now = time.time()
        for _ in range(100):
            ret = sess.run(interpolated_points)
        print(time.time() - now)
        print(ret.shape, ret.dtype)

    
    
