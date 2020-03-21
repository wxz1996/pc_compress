import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
devox_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_devoxelize.so'))

# sys.path.append('../voxelization')
from tf_vox import avg_voxel, voxelize, group_voxel
sys.path.append('../../utils')
from plyfile import PlyData, PlyElement
# tf.enable_eager_execution()

def trilinear_devoxelize(coords, features, resolution, is_training=True):
    '''
    Function: trilinear devoxelization (forward)
          Args:
            r        : voxel resolution
            trainig  : whether is training mode
            coords   : the coordinates of points, FloatTensor[b, 3, n]
            features : features, FloatTensor[b, c, s], s = r ** 3
          Return:
            outs : outputs, FloatTensor[b, c, n]
            inds : the voxel coordinates of point cube, IntTensor[b, 8, n]
            wgts : weight for trilinear interpolation, FloatTensor[b, 8, n]
    '''
    return devox_module.trilinear_devoxelize(coords,features,resolution,is_training)

@tf.RegisterGradient('TrilinearDevoxelize')
# 第一个参数为正向op,第二个参数为各个输出的梯度
def _trilinear_devoxelize_grad(op, grad_outs, grad_inds, grad_wgts):
    '''
    Function: trilinear devoxelization (backward)
          Args:
            grad_y  : grad outputs, FloatTensor[b, c, n]
            indices : the voxel coordinates of point cube, IntTensor[b, 8, n]
            weights : weight for trilinear interpolation, FloatTensor[b, 8, n]
            r       : voxel resolution
          Return:
            grad_x     : grad inputs, FloatTensor[b, c, s], s = r ** 3
    '''
    resolution = op.get_attr("resolution")
    indices = op.outputs[1]
    weights = op.outputs[2]
    return [None, devox_module.trilinear_devoxelize_grad(grad_outs, indices, weights, resolution)] # op有多个输入，但是有的输入不需要梯度



def write_ply(tensor, name):
    np.savetxt(name, np.squeeze(tensor.numpy().transpose(0, 2, 1)))
    len = tensor.numpy().shape[2]
    file = os.path.join('.', name)
    f = open(file, "r+")
    lines = [line.lstrip().rstrip().replace('  ', ' ') for line in f]
    vertex_nums, face_nums, _ = lines[1].split()
    f = open(file, "w+")
    f.seek(0)
    head = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + str(len) + "\nproperty float x\nproperty float y\nproperty float z\nelement face " + str(0) + "\nproperty list uchar int vertex_indices\nend_header\n"
    f.write(head)
    for line in lines[:]:
        f.write(line + "\n")
    f.close()

if __name__ == '__main__':
    import numpy as np
    import time

    for file in os.listdir('../../data'):
        file = os.path.join('../../data',file)
        plydata = PlyData.read(file)
        a_data = []
        for i in range(plydata['vertex'].count):
            line = [plydata['vertex']['x'][i], plydata['vertex']['y'][i], plydata['vertex']['z'][i]]
            a_data.append(line)
        pc = np.array(a_data)
    coords = pc[:, :3]
    pts = tf.expand_dims(tf.constant(pc),axis=0)
    pts = tf.transpose(pts,perm=[0, 2, 1])
    coords = tf.expand_dims(tf.constant(coords),axis=0)
    coords = tf.transpose(coords, perm=[0, 2, 1])
    vox_pc, norm_pc = voxelize(coords,8)
    write_ply(norm_pc,'vox_orl.ply')
    # np.savetxt('vox.ply',np.squeeze(vox_pc.numpy().transpose(0,2,1)))
    # write_ply(vox_pc, 'vox_pc.ply')
    vox_pc = tf.cast(vox_pc, tf.int32)
    res = 8
    # pts = tf.tile(pts,multiples=[4, 1, 1])
    # vox_pc = tf.tile(vox_pc,multiples=[4, 1, 1])
    voxel_feature, indices, counts = avg_voxel(res, pts, vox_pc)
    print(indices)
    print(counts)
    print(voxel_feature[0][1]) # [b, c, s]
    print(tf.expand_dims(voxel_feature[0],axis=0).shape)
    write_ply(tf.expand_dims(voxel_feature[0],axis=0), 'vox_pc_1.ply')
    # np.savetxt('vox_out_2.ply', np.squeeze(out.numpy().transpose(0, 2, 1)))
    print(norm_pc.shape)
    print(voxel_feature.shape)
    outs, idn, wgts = trilinear_devoxelize(norm_pc, voxel_feature, 8)
    write_ply(outs, 'out.ply')
    # out = np.squeeze(out1.numpy().transpose(0,2,1))
    # np.savetxt('out.ply',out)
    # print(idn)
    # print(wgts)
    # print(tf.reshape(out, (b, c, res, res, res)))
    # err = tf.test.compute_gradient_error(out,(1,6,64), out1,(1,6,128))



