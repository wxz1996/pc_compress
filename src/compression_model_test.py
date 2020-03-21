# coding=utf-8
import tensorflow as tf
import tensorflow_compression as tfc
import os
import sys
import math
import numpy as np
# tf.enable_eager_execution()
from collections import namedtuple
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = '/home/wenxuanzheng/pc_compression/pc_compression'
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import part_dataset
from transform_nets import input_transform_net
import tf_utils

def py_func_decorator(output_types=None, output_shapes=None, stateful=True, name=None):
    def decorator(func):
        def call(*args, **kwargs):
            return tf.contrib.framework.py_func(
                func=func,
                args=args, kwargs=kwargs,
                output_types=output_types, output_shapes=output_shapes,
                stateful=stateful, name=name
            )
        return call
    return decorator

def from_indexable(iterator, output_types, output_shapes=None, num_parallel_calls=None, stateful=True, name=None):
    ds = tf.data.Dataset.range(len(iterator))

    @py_func_decorator(output_types, output_shapes, stateful=stateful, name=name)
    def index_to_entry(index):
        return iterator[index]
    return ds.map(index_to_entry, num_parallel_calls=num_parallel_calls)

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

def input_fn(features, batch_size, preprocess_threads,  repeat=True, prefetch_size=1):
    with tf.device('/cpu:0'):
        # 需要iter对象
        # dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = from_indexable(features, output_types=tf.float32,output_shapes=[2048, 3], num_parallel_calls=preprocess_threads)
        if repeat:
            dataset = dataset.shuffle(buffer_size=len(features))
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
        # 流水线，一边生成一边使用
        dataset = dataset.prefetch(buffer_size=prefetch_size)
    return dataset

def model_fn(features, labels, mode, params):
    '''
    :param features:  batch_features from input_fn
    :param labels:  batch_labels from input_fn
    :param mode:    An instance of tf.estimator.ModeKeys
    :param params:  Additional configuration
    :return:
    '''
    #del para
    del labels
    if params.get('decompress') is None:
        params['decompress'] = False
    # if params.decompression:
    #     assert mode == tf.estimator.ModeKeys.PREDICT, 'Decompression must use prediction mode'

    params = namedtuple('Struct', params.keys())(*params.values())
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    num_points = (params.batch_size * params.num_points)
    pc = features
    bn_decay = get_bn_decay(tf.train.get_global_step())
    learning_rate = get_learning_rate(tf.train.get_global_step())
    tf.summary.scalar('bn_decay', bn_decay)
    tf.summary.scalar('learning_rate', learning_rate)

    # ============= encoder =============
    nasmples = params.knn
    y = pc_encoder(pc, nasmples, is_training=training, bn_decay=bn_decay)
    entropy_bottleneck = tfc.EntropyBottleneck()
    y_tilde, likelihoods = entropy_bottleneck(y, training=True)
    # ============= decoder =============
    x_tilde = pc_decoder(y_tilde, is_training=training, bn_decay=bn_decay)

    # number of bits divided by number of points
    train_bpp = tf.reduce_sum(tf.log(likelihoods)) / (-np.log(2) * num_points)
    # 预测模式直接返回结果
    if mode == tf.estimator.ModeKeys.PREDICT:
        string = entropy_bottleneck.compress(y)
        predictions = {
            'x_tilde': x_tilde,
            'likelihoods': likelihoods,
            'y_tilde': y_tilde
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions) # 生成predict字典
    # 训练和评估
    losses = tf_utils.get_loss(x_tilde, pc)
    rd_loss = losses + params.lmbda * train_bpp
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
    nn_dis, idx_batch = tf_utils.get_knn(point_cloud, nasmples)
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    idx_batch = tf.cast(idx_batch, dtype=tf.int32)
    latent_feature = {}
    # con_point = tf.concat([point_cloud, cov_batch], axis=2)
    # encoder_input = tf.expand_dims(con_point, -1)  # (32 2048 3 1)
    encoder_input = tf.expand_dims(point_cloud, -1)  # (32 2048 3 1)
    # (32, 2048, 1, 64)
    net = tf_utils.conv2d(encoder_input, 64, [1, 3],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='mlp_1', bn_decay=bn_decay)
    # (32, 2048, 1, 64)
    net = tf_utils.conv2d(net, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='mlp_2', bn_decay=bn_decay)
    # (32, 2048, 1, 64)
    net = tf_utils.conv2d(net, 64, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='mlp_3', bn_decay=bn_decay)
    net = tf_utils.conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='mlp_4', bn_decay=bn_decay)

    net = tf_utils.conv2d(net, 1024, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope='mlp_5', bn_decay=bn_decay)
    global_feat = tf_utils.max_pool2d(net, [num_point, 1],
                                      padding='VALID', scope='maxpool')
    net = tf.reshape(global_feat, [batch_size, -1])
    return net

def pc_decoder(y_tilde, is_training, bn_decay):
    # UPCONV Decoder
    batch_size = y_tilde.get_shape()[0].value
    net = tf_utils.fully_connected(y_tilde, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_utils.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_utils.fully_connected(net, 2048 * 3, activation_fn=None, scope='fc3')
    net = tf.reshape(net, (batch_size, 2048, 3))
    return net


if __name__=='__main__':
    tf.enable_eager_execution()
    TRAIN_DATASET = part_dataset.PartDataset(
        root='/data/dataset/shapenetcore_partanno_segmentation_benchmark_v0', npoints=2048,
        classification=False, class_choice=None, split='trainval')
    print('=============')
    print(input_fn(TRAIN_DATASET,2,8,repeat=True,prefetch_size=6))

