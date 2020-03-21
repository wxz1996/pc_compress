import tensorflow as tf
# tf.enable_eager_execution()
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', 'tf_ops/grouping'))
from tf_group import group_point, knn_points
sys.path.append(os.path.join(BASE_DIR, '..', 'tf_ops/pc_distance'))
import tf_nn_distance
sys.path.append(os.path.join(BASE_DIR, '..', 'tf_ops/3d_interpolation'))
import tf_interpolate
# sys.path.append(os.path.join(BASE_DIR, '..', 'tf_ops/inver_knn'))
# import reverse_knn

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    '''Helper to create a Varible stored on CPU memory
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    '''
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=True)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.
      Note that the Variable is initialized with a truncated normal distribution.
      A weight decay is added only if one is specified.

      Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        use_xavier: bool, whether to use xavier initializer

      Returns:
        Variable Tensor
      """
    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels] # 1 in out
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.get_shape()[-1].value
      elif data_format=='NCHW':
        num_in_channels = inputs.get_shape()[1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn',
                                        data_format=data_format)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs



def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     data_format='NHWC',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
    """ 2D convolution transpose with non-linear operation.

    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable

    Returns:
      Variable tensor

    Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_output_channels, num_in_channels]  # reversed to conv2d
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride

        # from slim.convolution2d_transpose
        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size

            if padding == 'VALID' and dim_size is not None:
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        # caculate output shape
        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                                         [1, stride_h, stride_w, 1],
                                         padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn',
                                            data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[num_input_units, num_outputs],
                                              use_xavier=use_xavier,
                                              stddev=stddev,
                                              wd=weight_decay)
        outputs = tf.matmul(inputs, weights)
        biases = _variable_on_cpu('biases', [num_outputs],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

def conv2d_nobias(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           is_dist=False):
    """ 2D convolution with non-linear operation.

      Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: a list of 2 ints
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

      Returns:
        Variable tensor
      """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('weights',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)

        if bn:
            outputs = batch_norm_for_conv2d(outputs, is_training,
                                            bn_decay=bn_decay, scope='bn',
                                            data_format=data_format)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D max pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
    """ 2D avg pooling.

    Args:
      inputs: 4-D tensor BxHxWxC
      kernel_size: a list of 2 ints
      stride: a list of 2 ints

    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool(inputs,
                                 ksize=[1, kernel_h, kernel_w, 1],
                                 strides=[1, stride_h, stride_w, 1],
                                 padding=padding,
                                 name=sc.name)
        return outputs


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
    """ Batch normalization on 2D convolutional maps.

    Args:
        inputs:      Tensor, 4D BHWC input maps
        is_training: boolean tf.Varialbe, true indicates training phase
        bn_decay:    float or float tensor variable, controling moving average weight
        scope:       string, variable scope
        data_format: 'NHWC' or 'NCHW'
    Return:
        normed:      batch-normalized maps
    """
    return batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay, data_format)


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
    '''
    Args:
        :param inputs: Tensor, k-D input ... x C could be BC or BHWC or BDHWC
        :param is_training: boolean tf.Varialbe, true indicates training phase
        :param scope: string, variable scope
        :param moments_dims_unused: a list of ints, indicating dimensions for moments calculation
        :param bn_decay: float or float tensor variable, controling moving average weight
        :param data_format: 'NHWC' or 'NCHW'
    return:
        normed: batch-normalized maps
    '''
    bn_decay = bn_decay if bn_decay is not None else 0.9
    return tf.contrib.layers.batch_norm(inputs,
                                        center=True,
                                        scale=True,
                                        is_training=is_training,
                                        decay=bn_decay,
                                        updates_collections=None,
                                        scope=scope,
                                        data_format=data_format)

def batch_norm_for_fc(inputs, is_trining, bn_decay, scope):
    '''
    Args:
        :param inputs: Tensor, 2D BxC input
        :param is_trining: boolean tf.Variable, true indiacate teaini
        :param bn_decay:
        :param scope:
    return:
        normed: batch-normalized maps
    '''
    return batch_norm_template(inputs, is_trining, scope, [0,], bn_decay)

def cuda_maxpooling(input, nn_idx):
    group_feature = group_point(input, nn_idx)
    group_feature = tf.reduce_max(group_feature,axis=2)
    # group_feature = tf.reduce_min(group_feature,axis=2)
    return group_feature

def knn_feature(input, nn_idx):
    group_feature = group_point(input, nn_idx)
    return group_feature


def get_loss(pred, pointclouds_pl):
    with tf.variable_scope('loss') as LossEvaluation:
        dists_forward, _, dists_backward, _ = tf_nn_distance.nn_distance(pred, pointclouds_pl)
        # loss = tf.reduce_mean(dists_forward + dists_backward)
        loss = (tf.reduce_mean(tf.sqrt(dists_forward)) + tf.reduce_mean(tf.sqrt(dists_backward)))/2
        # loss = tf_util_loss.nn_distance_fold(pred, pointclouds_pl)
        return loss

def get_knn(point_cloud, nsample):
    val, idx = knn_points(nsample, point_cloud, point_cloud)
    return val, idx

def node_group_point(pc, mask, min_idx):
    batch_size = mask.shape[0]
    point_num = mask.shape[1]
    node_num = mask.shape[2]
    max_idx = tf.Variable(tf.zeros((batch_size,node_num), dtype=tf.int32))
    max_val = tf.Variable(tf.constant(-1000, shape=(batch_size,node_num,384), dtype=tf.float32))
    sum_pc = tf.reduce_sum(pc, axis=2)
    for i in range(batch_size):  # 32
        for j in range(point_num):  # 2048*3
            tf.cond(sum_pc[i,j] > tf.reduce_sum(max_val[i,min_idx[i,j]]), lambda : (max_val[i,min_idx[i,j]].assign(pc[i,j]),max_idx[i,min_idx[i,j]].assign(j)), lambda : (max_val,max_idx))
            # print(tf.reduce_sum(max_val[i,2]))
    return max_val

# def reverse_knn_group(points, idx, som_num):
#     pooling_node = reverse_knn.reverse_knn(points, idx, som_num)
#     return pooling_node

def node_interpolate(xyz1, xyz2, points, scope):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points: (batch_size, ndataset1, nchannel1) TF tensor
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = tf_interpolate.three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = tf_interpolate.three_interpolate(points, idx, weight)
        new_point = tf.concat(axis=1, values=[interpolated_points, points]) # B,ndataset1,nchannel
        return new_point

def top_k(feature, k):
    feature = tf.transpose(feature, perm=[0, 3, 2, 1])
    feature,_ = tf.nn.top_k(feature, k)
    feature = tf.transpose(feature, perm=[0, 3, 2, 1])
    feature = tf.squeeze(feature)
    feature = tf.expand_dims(feature, -1)  # 32 3 2048 1
    return feature

def attn_layer(input_feature, output_dim, neighbors_idx, activation, in_dropout=0.0, coef_dropout=0.0, is_training=None, bn_decay=None, layer='', i=0, is_dist=False, k=3):
    batch_size = input_feature.get_shape()[0].value
    num_dim = input_feature.get_shape()[-1].value
    input_feature = tf.squeeze(input_feature)
    if batch_size == 1:
        input_feature = tf.expand_dims(input_feature, 0)

    input_feature = tf.expand_dims(input_feature, axis=-2)

    # 32 2048 1 16
    new_feature = conv2d_nobias(input_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope=layer + '_newfea_conv_head_' + str(i), bn_decay=bn_decay)
    nn_features = tf.squeeze(input_feature)
    neighbors = group_point(nn_features, neighbors_idx)
    input_feature_tiled = tf.tile(input_feature, [1, 1, k, 1])
    edge_feature = input_feature_tiled - neighbors
    edge_feature = conv2d(edge_feature, output_dim, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope=layer + '_edgefea_' + str(i), bn_decay=bn_decay)
    self_attention = conv2d(new_feature, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope=layer + '_self_att_conv_head_' + str(i), bn_decay=bn_decay)

    neibor_attention = conv2d(edge_feature, 1, [1, 1], padding='VALID', stride=[1, 1], bn=True, is_training=is_training, scope=layer + '_neib_att_conv_head_' + str(i), bn_decay=bn_decay)

    logits = self_attention + neibor_attention
    logits = tf.transpose(logits, [0, 1, 3, 2])
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
    vals = tf.matmul(coefs, edge_feature)
    if is_dist:
        ret = activation(vals)
    else:
        ret = tf.contrib.layers.bias_add(vals)
        ret = activation(ret)
    return ret, coefs, edge_feature


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.nn.top_k(neg_adj, k=k)
    return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    point_cloud_shape = point_cloud.get_shape()
    batch_size = point_cloud_shape[0].value
    num_points = point_cloud_shape[1].value
    num_dims = point_cloud_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    print(point_cloud_central)
    print(point_cloud_neighbors - point_cloud_central)
    print(edge_feature)
    return edge_feature

if __name__ == '__main__':
    # node_idx = tf.constant(([[0,0,1,0,1,0,1,0,0],
    #                          [0,0,1,1,0,0,0,0,0],
    #                          [0,0,0,0,1,1,0,0,1]],
    #                         [[1,0,1,1,1,0,1,0,0],
    #                          [0,1,0,0,0,0,0,1,0],
    #                          [1,0,0,1,0,0,0,1,0]]))
    points = tf.random_normal((8,20,3))
    adj_matrix = pairwise_distance(points)
    nn_idx = knn(adj_matrix, k=4)
    edge_feature = get_edge_feature(points, nn_idx=nn_idx, k=4)

    # print(edge_feature)
    # masked_point = tf.transpose(points, [0,2,1])
    # tf.enable_eager_execution()
    # pc = tf.random_normal((2,6,1,3))
    # print(pc)
    # pc = tf.transpose(pc,perm=[0,3,2,1])
    # top_k,_ = tf.nn.top_k(pc, 3)
    # top_k = tf.transpose(top_k, perm=[0,3,2,1])
    # print(top_k)
    # exit(-1)
    # min_idx = tf.convert_to_tensor(np.random.randint(0,64, size=(32,2048*3), dtype=np.int32))
    # # mask = tf.convert_to_tensor(np.random.randint(0,2, size=(32,2048*3,64), dtype=np.int32))
    # # pc = tf.reduce_sum(pc, axis=2)
    # print(pc)
    # # print(mask)
    # print(min_idx)
    # max_val = reverse_knn_group(pc, min_idx, 64)
    # print(max_val)
    # pc = tf.random_normal((2,2048,3))
    # adj_matrix = pairwise_distance(pc)
    # print(adj_matrix)






