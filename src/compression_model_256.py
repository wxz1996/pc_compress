# coding=utf-8
import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc
import os
import sys
import math
import numpy as np
# tf.enable_eager_execution()
from collections import namedtuple
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
import part_dataset
from transform_nets import input_transform_net
import pc_utils
import tf_util

sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/pc_distance'))
import tf_nn_distance
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/emd'))
import tf_auctionmatch
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
import tf_sampling

def get_cd_loss(pred, pc):
    """ pred: BxNx3,
        label: BxN, """
    dists_forward,_,dists_backward,_ = tf_nn_distance.nn_distance(pred, pc)
    loss = (tf.reduce_mean(tf.sqrt(dists_forward)) + tf.reduce_mean(tf.sqrt(dists_backward)))/2
    return loss

def get_emd_loss(pred, pc, radius):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, pc)
    matched_out = tf_sampling.gather_point(pc, matchl_out)
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    dist_norm = dist / radius
    emd_loss = tf.reduce_mean(dist_norm)
    return emd_loss

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        0.0001,
                        batch , # global_step 当前迭代次数
                        10000,
                        0.7,
                        staircase = True) # global_step / decay_steps始终取整数
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                        0.5,
                        batch,
                        20000,
                        0.5,
                        staircase=True)
    bn_decay = tf.minimum(0.99, 1 - bn_momentum)
    return bn_decay

def input_fn(features, batch_size, preprocess_threads,  repeat=True, prefetch_size=1, is_train = True):
    with tf.device('/cpu:0'):
        # make a dataset from a numpy array
        dataset = tf.data.Dataset.from_generator(lambda: iter(features), tf.float32, tf.TensorShape([256, 3]))
        # create the iterator
        if repeat:
            dataset = dataset.shuffle(buffer_size=len(features))
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        if is_train:
            dataset = dataset.map(lambda x: pc_utils.rotate_point_cloud_and_gt(x, None),
                                  num_parallel_calls=preprocess_threads)
            dataset = dataset.map(
                lambda x: pc_utils.random_scale_point_cloud_and_gt(x, None, scale_low=0.8, scale_high=1.2),
                num_parallel_calls=preprocess_threads)
            dataset = dataset.map(lambda x: pc_utils.jitter_perturbation_point_cloud(x, sigma=0.005, clip=0.015),
                                  num_parallel_calls=preprocess_threads)
            dataset = dataset.map(lambda x: pc_utils.rotate_perturbation_point_cloud(x, angle_sigma=0.03, angle_clip=0.09),
                                  num_parallel_calls=preprocess_threads)
        # 流水线，一边生成一边使用
        dataset = dataset.prefetch(buffer_size=prefetch_size)
        # 使用创建的数据集来构造一个Iterator实例以遍历数据集
    return dataset.make_one_shot_iterator().get_next()
    # tf2.0
    # return dataset

def model_fn(features, labels, mode, params):
    '''
    :param features:  batch_features from input_fn
    :param labels:  batch_labels from input_fn
    :param mode:    An instance of tf.estimator.ModeKeys
    :param params:  Additional configuration
    :return:
    '''
    if params.get('decompress') is None:
        params['decompress'] = False
    params = namedtuple('Struct', params.keys())(*params.values())
    del labels
    if params.decompress:
        assert mode == tf.estimator.ModeKeys.PREDICT, 'Decompression must use prediction mode'
        entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
        y_tilde = entropy_bottleneck.decompress(features, [256], channels=256)  # B*N
        x_hat = pc_decoder(y_tilde, params.batch_size, is_training=False, bn_decay=False)
        predictions = {
            'y_tilde': y_tilde,
            'x_hat': x_hat
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)


    training = (mode == tf.estimator.ModeKeys.TRAIN)
    # Get training patch from dataset.
    # num_points = (params.batch_size * params.num_points)
    batch_size = int(features.shape[0])
    num_points = int(features.shape[1])
    pc = features
    bn_decay = get_bn_decay(tf.train.get_global_step())
    learning_rate = get_learning_rate(tf.train.get_global_step())
    tf.summary.scalar('bn_decay', bn_decay)
    tf.summary.scalar('learning_rate', learning_rate)

    # ============= encoder =============
    y = pc_encoder(pc, params.knn, is_training=training, bn_decay=bn_decay)

    # ============= bottleneck layer =============
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)

    # ============= decoder =============
    x_tilde = pc_decoder(y_tilde, params.batch_size, is_training=training, bn_decay=bn_decay)

    # number of bits divided by number of points
    train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * int(num_points))

    # 预测模式直接返回结果
    if mode == tf.estimator.ModeKeys.PREDICT:
        string = entropy_bottleneck.compress(y)
        predictions = {
            'string': string,
            'x_tilde': x_tilde,
            'y_tilde': y_tilde
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)  # 生成predict字典

    # 训练和评估
    losses = get_emd_loss(x_tilde, pc, 1)
    rd_loss = params.lmbda * train_bpp + losses
    # tf.summary.scalar('likelihoods',likelihoods)
    tf.summary.scalar('loss', losses)
    tf.summary.scalar('rd_loss', rd_loss)
    tf.summary.scalar('bpp', train_bpp)

    if mode == tf.estimator.ModeKeys.TRAIN:
        main_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        main_step = main_optimizer.minimize(rd_loss,  global_step=tf.train.get_global_step())

        aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        aux_step = aux_optimizer.minimize(entropy_bottleneck.losses[0])

        train_op = tf.group(main_step, aux_step, entropy_bottleneck.updates[0])

        return tf.estimator.EstimatorSpec(mode, loss=rd_loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        summary_hook = tf.train.SummarySaverHook(
            save_steps=5,
            output_dir=os.path.join(params.checkpoint_dir, 'eval'),
            summary_op=tf.summary.merge_all())
        return tf.estimator.EstimatorSpec(mode, loss=rd_loss, evaluation_hooks=[summary_hook])



def pc_encoder(point_cloud, nasmples, is_training, bn_decay=None):
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    with tf.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)

    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, -1)
    nn_dis, idx_batch = tf_util.get_knn(point_cloud, 12)

    # Encoder
    net = tf_util.conv2d(point_cloud_transformed, 64, [1, point_dim],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat_1 = tf_util.conv2d(net, 128, [1, 1],
                                  padding='VALID', stride=[1, 1],
                                  bn=True, is_training=is_training,
                                  scope='conv3', bn_decay=bn_decay)

    print('------------ convPN_1 ------------')
    point_feat = tf_util.conv2d(point_feat_1, 256, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv4', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(point_feat, 256, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv5', bn_decay=bn_decay)
    feature = tf.squeeze(point_feat, squeeze_dims=2)
    knn_feat = tf_util.cuda_maxpooling(feature, idx_batch)
    knn_feat = tf.expand_dims(knn_feat, axis=2)
    point_feat_2 = tf.concat([point_feat, knn_feat], axis=-1)  # 32 256 1 256

    print('------------ convPN_2 ------------')
    print(point_feat_2)
    point_feat = tf_util.conv2d(point_feat_2, 256, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv6', bn_decay=bn_decay)
    print(point_feat)
    point_feat = tf_util.conv2d(point_feat, 256, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv7', bn_decay=bn_decay)
    feature = tf.squeeze(point_feat, squeeze_dims=2)
    knn_feat = tf_util.cuda_maxpooling(feature, idx_batch)
    knn_feat = tf.expand_dims(knn_feat, axis=2)
    point_feat_3 = tf.concat([point_feat, knn_feat], axis=-1)  # 32 256 1 512
    mix_feature = tf.concat([point_feat_1, point_feat_2, point_feat_3], axis=-1)

    # ----------- maxpooling--------------
    global_feature = tf_util.max_pool2d(mix_feature, [num_point, 1], padding='VALID', scope='maxpool_1')
    net = tf.reshape(global_feature, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc00', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc01', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc02', bn_decay=bn_decay)
    print(net)
    net = tf.reshape(net, [batch_size, -1])
    print(net)
    return net

def pc_decoder(y_tilde, batch_size, is_training, bn_decay):
    # UPCONV Decoder
    net = tf.reshape(y_tilde, [batch_size, 1, 1, 256])
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256 * 3, activation_fn=None, scope='fc3')
    pc_fc = tf.reshape(net, (batch_size, -1, 3))
    return pc_fc


if __name__=='__main__':
    # tf.enable_eager_execution()
    # TRAIN_DATASET = part_dataset.PartDataset(
    #     root='/data/dataset/shapenetcore_partanno_segmentation_benchmark_v0', npoints=2048,
    #     classification=False, class_choice=None, split='trainval')
    print('=============')
    # print(input_fn(TRAIN_DATASET,2,8,repeat=True,prefetch_size=6))

