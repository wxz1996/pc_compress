import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
# import sparse
tf.enable_eager_execution()

def focal_loss(y_true, y_pred, gamma=2, alpha=0.95):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    # return -(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - ((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0)), -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def pc_to_tf(points, dense_tensor_shape):
    x = points
    x = tf.pad(x, [[0, 0], [1, 0]])
    print(x)
    st = tf.sparse.SparseTensor(x, tf.ones_like(x[:, 0]), dense_tensor_shape)
    print(st)
    return st

def process_x(x, dense_tensor_shape):
    x = tf.sparse.to_dense(x, default_value=0, validate_indices=False)
    x.set_shape(dense_tensor_shape)
    x = tf.cast(x, tf.float32)
    return x


if __name__ == '__main__':
    coords = np.random.randint(0, 64 - 1, size=(3, 4096))

    dense_tensor_shape = np.array([1, 64, 64, 64])
    points = tf.constant(coords,dtype=tf.int64)
    points = tf.transpose(points,(1,0))
    data = pc_to_tf(points,dense_tensor_shape)
    print(data)
    data = process_x(data,dense_tensor_shape)
    data = data.numpy()
    print(np.sum(data==1))

    coords = torch.from_numpy(coords)
    coo_metrix = torch.sparse_coo_tensor(coords, torch.ones(2048), size=(64, 64, 64))
    coo = coo_metrix.to_dense()
    coo = coo.numpy()
    print(np.sum(coo == 1))



    # points = np.array([[0, 1, 2],
    #                       [2, 1, 2],
    #                       [0, 1, 0],
    #                       [0, 1, 1],
    #                       [2, 1, 0],
    #                       [1, 1, 2]],dtype=np.int32)
    # print(points.T.shape)
    # x = sparse.COO(coords, 1, shape=((64,) * 3))
    #
    # y = sparse.tensordot(x, x, axes=((2, 0), (0, 1)))
    #
    # z = y.sum(axis=(0, 1, 2))
    # print(z.todense())
    #
    # data = np.array(x.todense())
    # print(np.sum(data == 1))

    n = 64
    ndims = 3
    nnz = 1000
    coords = np.random.randint(0, n - 1, size=(ndims, nnz))

    s = sparse.SparseArray.fromdense(coords)
    print(s)

    # print(coords)
    # data = np.random.random(nnz)
    # print(data)
    x = sparse.COO(coords, 1, shape=((n,) * ndims))
    # y = sparse.tensordot(x, x, axes=((3, 0), (1, 2)))
    # z = y.sum(axis=(0, 1, 2))
    data = np.array(x.todense())
    print(data.shape)

    print(np.sum(data == 1))


    # print(data)
    # print(data.shape)
    # x = tf.constant([[[[0.6, 0.3, 0.7],
    #                    [0.6, 0.3, 0.7],
    #                    [0.6, 0.3, 0.7]],
    #                   [[0.6, 0.3, 0.7],
    #                    [0.6, 0.3, 0.7],
    #                    [0.6, 0.3, 0.7]],
    #                   [[0.6, 0.3, 0.7],
    #                    [0.6, 0.3, 0.7],
    #                    [0.6, 0.3, 0.7]]]])
    # print(x.shape)
    # x = tf.random_normal((3,3,3))
    # y = np.random.randint(0,2,size=(3,3,3))
    # y = tf.constant([[[[1,0,1],
    #                    [1,0,1],
    #                    [1,0,1]],
    #                   [[1,0,1],
    #                    [1,0,1],
    #                    [1,0,1]],
    #                   [[1,0,1],
    #                    [1,0,1],
    #                    [1,0,1]]]])
    # x = tf.expand_dims(x,axis=0)
    # y = tf.expand_dims(y,axis=0)
    # print(focal_loss(data,x))
