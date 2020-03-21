# coding=utf-8
import tensorflow as tf
import scipy.sparse
from sklearn.neighbors import KDTree
import numpy as np
import math
import multiprocessing as multiproc
from functools import partial
tf.enable_eager_execution()

def edges2A(edges, n_nodes, mode='P', sparse_mat_type=scipy.sparse.csr_matrix):
    '''
    note: assume no (i,i)-like edge
    edges: <2xE>
    '''
    edges = np.array(edges).astype(int)

    data_D = np.zeros(n_nodes, dtype=np.float32)
    for d in range(n_nodes):
        data_D[ d ] = len(np.where(edges[0] == d)[0])   # compute the number of node which pick node_i as their neighbor

    if mode.upper() == 'M':  # 'M' means max pooling, which use the same graph matrix as the adjacency matrix
        data = np.ones(edges[0].shape[0], dtype=np.int32)
    elif mode.upper() == 'P':
        data = 1. / data_D[ edges[0] ]
    else:
        raise NotImplementedError("edges2A with unknown mode=" + mode)

    return sparse_mat_type((data, edges), shape=(n_nodes, n_nodes))


def build_graph_core(batch_data):
    try:
        points = batch_data # 2048*3
        n_points = points.shape[0]
        edges, dis, cov, idx = knn_search(points)
        edges_z = edges2A(edges, n_points, mode='M', sparse_mat_type=scipy.sparse.csr_matrix)
        dis = np.asarray(dis)[:,1:]
        dis = np.reshape(dis, -1)
        return edges.T, edges_z, dis, cov, idx
    except KeyboardInterrupt:
        exit(-1)

def build_graph(point_cloud):
    batch_size = point_cloud.shape[0]
    num_point = point_cloud.shape[0]
    point_dim = point_cloud.shape[1]
    # point_cloud = point_cloud.eval()
    batch_graph = []
    Cov = np.zeros((batch_size, num_point, 9))
    nn_idx = np.zeros((batch_size, num_point, 17))

    # pool = multiproc.Pool(2) # 进程池,保证池中只有两个进程
    # pool_func = partial(build_graph_core) # 先传一部分参数
    # rets = pool.map(pool_func, point_cloud)
    # pool.close()
    rets = build_graph_core(point_cloud)
    count = 0

    for ret in rets:
        point_graph, _, _, cov,graph_idx = ret
        batch_graph.append(point_graph)
        # Cov[count,:,:] = tf.convert_to_tensor(cov)
        Cov[count,:,:] = cov
        nn_idx[count,:,:] = graph_idx
        count += 1
    del rets
    return batch_graph, Cov, nn_idx

def knn_search(point_cloud, knn=16, metric='minkowski', symmetric=True):
    '''
    Args:
        :param point_cloud: Nx3
        :param knn: k
    return:
    '''
    assert(knn > 0)
    #num_point = point_cloud.get_shape()[0].value
    num_point = point_cloud.shape[0]
    kdt = KDTree(point_cloud, leaf_size=30, metric=metric)

    dis, idx = kdt.query(point_cloud, k=knn+1, return_distance=True)
    cov = np.zeros((num_point, 9))
    # Adjacency Matrix
    adjdict = dict()
    for i in range(num_point):
        nn_index = idx[i] # nearest point index
        # compute local covariance matrix 3*3=>1*9
        cov[i] = np.cov(np.transpose(point_cloud[nn_index[1:]])).reshape(-1)
        for j in range(knn):
            if symmetric:
                adjdict[(i, nn_index[j+1])] = 1
                adjdict[(nn_index[j+1], i)] = 1
            else:
                adjdict[(i, nn_index[j + 1])] = 1
    edges = np.array(list(adjdict.keys()), dtype=int).T
    return edges, dis, cov, idx


def GridSamplingLayer(batch_size, meshgrid):
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

if __name__=='__main__':
    # meshgrid = [[-0.3,0.3,45],[-0.5,0.5,45]]
    # out = GridSamplingLayer(3, meshgrid)
    # print('meshgrid; ', out)


    pcd = np.random.random((2,2048,3))
    batch_graph, Cov, idx = build_graph(pcd)
    print(batch_graph, Cov, idx.shape)
    # pcd2 = tf.Variable(tf.random_uniform([2,2048,1,3]))
    # idx = tf.to_int32(idx)
    # nn_point = tf.Variable(tf.zeros((2, 2048, 17,1 ,3)))
    # for i in range(2):
    #     for j in range(2048):
    #         nn_point[i,j].assign(tf.gather(pcd2[i],idx[i, j, :]))
    # print(tf.reduce_max(nn_point,axis=2))

    # tf.enable_eager_execution()
    # pcd = np.random.random((2, 2048, 3))
    # batch_graph, Cov, idx = build_graph(pcd)
    # pcd2 = np.random.randint(0, 100, (2, 2048, 64))
    # idx = tf.to_int32(idx)
    # nn_point = np.zeros((2, 2048, 17, 64))
    # nn_point[0:2, 0:2048] = tf.gather(pcd2[0:2], idx[0:2, 0:2048, :]).numpy()
    # print(tf.reduce_max(nn_point,axis=2))
    # nn_point2 = np.zeros((2, 2048, 17, 64))
    # for i in range(2):
    #     for j in range(2048):
    #         nn_point2[i:j] = tf.gather(pcd2[i],idx[i, j, :]).numpy()
    # print(tf.reduce_max(nn_point2,axis=2))


    #print(tf.cast(idx[0][0][1],dtype=tf.int32))
    #print(pcd[tf.cast(idx[0][0],dtype=tf.float32)])
    #print(batch_graph)
    exit(-1)
    indices=[]
    values=[]
    for n,seq in enumerate(batch_graph[0]):
        indices.append(zip([n]*len(seq),range(len(seq))))
        values.append(seq)
    index = batch_graph[0].nonzero()
    print(index)
    #print(tf.contrib.layers.dense_to_sparse(batch_graph[1]))
    nn_point = np.zeros((2048,16))
    for i in range(3):
        idx = index[0] == i
        ele = index[1][idx]
        # ele = index[1][False]
        #print(ele)
        rand_idx = np.random.choice(len(ele), 16, replace=False)
        #print(rand_idx)
        ele = ele[rand_idx]
        nn_point[i, :] = ele
        #print(nn_point.shape)
        #print(nn_point)
    nn_point = nn_point.astype(np.int)
    pcd = pcd.astype(np.int)
    nn_point = pcd[0][nn_point]
    nn_point = np.expand_dims(nn_point,axis=0)
    print('---------------')
    print(nn_point)
    print(nn_point.shape)
    pcd = np.expand_dims(pcd,axis=2)
    print(pcd)
    print(pcd.shape)
    nn_point = np.concatenate(nn_point, pcd)
    nn_point = tf.convert_to_tensor(nn_point)
    nn_point = tf.reduce_max(nn_point,axis=1)
    nn_point = tf.maximum(nn_point,tf.squeeze(pcd,axis=0))
    print(nn_point)
    #print(nn_point)
    #print(pcd[0][nn_point[0][15]])
    np.maximum(pcd[0][nn_point],pcd)
    #ele = index[1][idx]