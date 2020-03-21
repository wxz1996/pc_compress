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

sys.path.append('../voxelization')
from tf_vox import group_voxel, voxelize, avg_voxel
sys.path.append('../../utils')
from tf_devox import trilinear_devoxelize
import tf_util
from plyfile import PlyData, PlyElement

sys.path.append(os.path.join(BASE_DIR, '..', 'tf_ops/pc_distance'))
import tf_nn_distance
tf.enable_eager_execution()

def get_cd_loss(pred, pc):
    """ pred: BxNx3,
        label: BxN, """
    dists_forward,_,dists_backward,_ = tf_nn_distance.nn_distance(pred, pc)
    # loss = tf.reduce_mean(dists_forward+dists_backward)
    loss = (tf.reduce_mean(tf.sqrt(dists_forward)) + tf.reduce_mean(tf.sqrt(dists_backward)))/2
    return loss

def write_ply(tensor, name):
    np.savetxt(name, np.squeeze(tensor.numpy().transpose(0, 2, 1)))
    len = tensor.numpy().shape[2]
    file = os.path.join('.', name)
    f = open(file, "r+")
    lines = [line.lstrip().rstrip().replace('  ', ' ') for line in f]
    vertex_nums, face_nums, _ = lines[1].split()
    f = open(file, "w+")
    f.seek(0)
    head = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + '2048' + "\nproperty float x\nproperty float y\nproperty float z\nelement face " + str(0) + "\nproperty list uchar int vertex_indices\nend_header\n"
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
    # pts = tf.transpose(pts,perm=[0, 2, 1])

    # pc_input = tf.placeholder(tf.float32, shape=(2, 2048, 3))
    #
    pts = np.tile(pts,reps=(1,1,1))
    print(pts.shape) # (1, 2048, 3)
    pts = tf.constant(pts)
    voxel_feature = tf_util.pvcnn((pts,pts), scope='pvcnn_1',num_output_channels=3,
                  resolution=64, bn=True, is_training=True,
                  bn_decay=0.2)
    print(voxel_feature.shape)
    write_ply(tf.transpose(voxel_feature,perm=(0,2,1)), 'vox_pc.ply')
    # # my_output = tf.multiply(pc_input, 0.01)
    #
    # loss = tf.Variable(tf.random_normal(shape=[1]))
    # my_opt = tf.train.GradientDescentOptimizer(0.02)
    # train_step = my_opt.minimize(loss)
    #
    #
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     for i in range(1000):
    #         sess.run(train_step, feed_dict={pc_input: pts})
    #         print('Loss = ' + str(sess.run(loss, feed_dict={pc_input: pts})))
    exit(-1)




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
    indices, counts = group_voxel(res, pts, vox_pc)
    print(indices)
    print(counts)
    pts = tf.transpose(pts,perm=(0,2,1))
    out = avg_voxel(res,pts,indices,counts)
    out = tf.transpose(out,perm=(0,2,1))
    write_ply(out, 'vox_pc.ply')
    print(out.shape)
    print(norm_pc.shape)
    # feature_list = []


    outs, idn, wgts = trilinear_devoxelize(norm_pc, out, 8)
    print(outs.shape)
    print(idn)
    write_ply(outs, 'out.ply')
