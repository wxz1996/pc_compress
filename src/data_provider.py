#!/usr/bin/env python

################################################################################
### Init
################################################################################
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

import os
from os.path import join, basename, split, splitext
from os import makedirs
from glob import glob
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import functools
from collections import Counter;
import os
import open3d as o3d
import random
import math
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement


################################################################################
### Definitions
################################################################################

def pc_normalize(pc):
    """pc: NxC, return NxC"""
    l = pc.shape[0]
    centriod = np.mean(pc, axis=0)
    pc = pc - centriod
    furthest_distance = np.max(np.sqrt(np.sum(pc**2, axis=1))) 
    pc = pc/furthest_distance
    return pc, centriod, furthest_distance

def gen_meta(centroid_list, furthest_distance_list, pcQueue1, pcQueue2, pcQueue3):
    nor_pc_1 = []
    nor_pc_2 = []
    nor_pc_3 = []
    for pc in tqdm(pcQueue1):
        pc, centroid, furthest_distance = pc_normalize(pc)
        centroid_list.append(centroid)
        furthest_distance_list.append(furthest_distance)
        nor_pc_1.append(pc)
    for pc in tqdm(pcQueue2):
        pc, centroid, furthest_distance = pc_normalize(pc)
        centroid_list.append(centroid)
        furthest_distance_list.append(furthest_distance)
        nor_pc_2.append(pc)
    for pc in tqdm(pcQueue3):
        pc, centroid, furthest_distance = pc_normalize(pc)
        centroid_list.append(centroid)
        furthest_distance_list.append(furthest_distance)
        nor_pc_3.append(pc)
    out_centroid = np.vstack(centroid_list[:])
    out_furthest_distance = np.vstack(furthest_distance_list[:])
    meta_matrix = np.hstack((out_centroid, out_furthest_distance))
    meta_matrix = meta_matrix.astype(np.float16)
    return meta_matrix,nor_pc_1,nor_pc_2,nor_pc_3

def pc_read(pc, point_nums):
    global num
    if pc.shape[0]>=point_nums:
        row_rand_array = np.arange(pc.shape[0])
        np.random.shuffle(row_rand_array)
        select_point = pc[0:point_nums]
    else:
        idx1 = np.random.randint(pc.shape[0], size=(point_nums-pc.shape[0]))
        idx2 = np.arange(pc.shape[0])
        idx = np.append(idx1, idx2)
        select_point = pc[idx,:]
    return select_point[:,:3]

def epsilon(data, MinPts):
    m, n = np.shape(data)
    xMax = np.max(data, 0)
    xMin = np.min(data, 0)
    eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps

def dbscan(points):
    my_eps = epsilon(points, 10)
    y_pred = DBSCAN(eps=my_eps).fit_predict(points)
    cluster_num = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    return cluster_num, y_pred

def is_division(points):
    ori_pc = points.points.values[:, :3]
    cluster_num, y_pred = dbscan(ori_pc)
    return cluster_num, y_pred

def traverse_recurse(pc, pcQueue_512, pcQueue_1024, pcQueue_2048):
    voxelgrid_id = pc.add_structure("voxelgrid", n_x=2, n_y=2, n_z=2)
    voxelgrid = pc.structures[voxelgrid_id]
    voxel_n = pc.add_scalar_field("voxel_n", voxelgrid_id=voxelgrid_id)
    splits = {x: PyntCloud(pc.points.loc[pc.points[voxel_n] == x]) for x in pc.points[voxel_n].unique()}
    for idx in range(8):
        point_nums = Counter(voxelgrid.voxel_n)[idx]
        try:
            if point_nums <= 2560:
                cluster_num, y_pred = is_division(splits[idx])
                if cluster_num > 1 and cluster_num < 10:
                    for cluster_idx in range(cluster_num):
                        cluster_pc = splits[idx].points.values[y_pred == cluster_idx]
                        # points = splits[idx].points.values[y_pred == idx]
                        pc_num = cluster_pc.shape[0]
                        if 30 <= pc_num <= 768:
                            print("cluster 0 - 768", pc_num)
                            pcQueue_512.append(pc_read(cluster_pc, 512))
                        elif pc_num <= 1536 and pc_num > 768:
                            print("cluster 768 - 1536", pc_num)
                            pcQueue_1024.append(pc_read(cluster_pc, 1024))
                        elif pc_num <= 2560 and pc_num > 1536:
                            print("cluster 1536 - 2560", pc_num)
                            pcQueue_2048.append(pc_read(cluster_pc, 2048))
                else:
                    if 30 <= point_nums <= 768:
                        print("0 - 768", point_nums)
                        pcQueue_512.append(pc_read(splits[idx].points.values[y_pred == 0], 512))
                    elif point_nums <= 1536 and point_nums > 768:
                        print("1536 - 768", point_nums)
                        pcQueue_1024.append(pc_read(splits[idx].points.values[y_pred == 0], 1024))
                    elif point_nums <= 2560 and point_nums > 1536:
                        print("2560 - 1536", point_nums)
                        pcQueue_2048.append(pc_read(splits[idx].points.values[y_pred == 0], 2048))
            else:
                traverse_recurse(splits[idx], pcQueue_512, pcQueue_1024, pcQueue_2048)
        except Exception:
            print('no point')


def process(source):
    pc_mesh = PyntCloud.from_file(source)
    pcQueue_512 = []
    pcQueue_1024 = []
    pcQueue_2048 = []
    traverse_recurse(pc_mesh, pcQueue_512, pcQueue_1024, pcQueue_2048)
    pc = pcQueue_512 + pcQueue_1024 + pcQueue_2048
    pc = np.vstack(pc[:])
    return pcQueue_512, pcQueue_1024, pcQueue_2048

def load_data(source_path, source_extension):
    num = 0
    paths = glob(join(source_path, '**', f'*{source_extension}'), recursive=True)
    files = [x[len(source_path) + 1:] for x in paths]
    files_len = len(files)
    assert files_len > 0
    source = join(source_path, files[0])
    return process(source)


################################################################################
### Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='mesh_to_pc.py',
        description='Converts a folder containing meshes to point clouds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--source',default='../eval_input', help='Source directory')
    parser.add_argument('--source_extension', help='Mesh files extension', default='.ply')
    args = parser.parse_args()

    assert os.path.exists(args.source), f'{args.source} does not exist'
    # assert not os.path.exists(args.dest), f'{args.dest} already exists'
    assert args.vg_size > 0, f'vg_size must be positive'
    assert args.n_samples > 0, f'n_samples must be positive'

    paths = glob(join(args.source, '**', f'*{args.source_extension}'), recursive=True)
    files = [x[len(args.source) + 1:] for x in paths]
    files_len = len(files)
    assert files_len > 0
    logger.info(f'Found {files_len} models in {args.source}')

    depth = 0
    with Pool() as p:
        process_f = functools.partial(process, args=args)
        list(tqdm(p.imap(process_f, files), total=files_len))

    logger.info(f'{files_len} models written to {args.dest}')
