import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import multiprocessing
import functools
from tqdm import tqdm
from pyntcloud import PyntCloud
from glob import glob


class PC:
    def __init__(self, points):
        self.points = points
        self.data = {}

    def __repr__(self):
        # 直接打印对象的时调用
        return f"<PC with {self.points.shape[0]} points>"

    def is_empty(self):
        return self.points.shape[0] == 0



def df_to_pc(df):
    points = df[['x', 'y', 'z']].values
    points = pc_normalize(points)
    # print(np.where(np.isinf(points)))
    return PC(points)


def pa_to_df(points):
    df = pd.DataFrame(data={
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2]
    }, dtype=np.float32)

    return df


def pc_normalize(pc):
    """pc: NxC, return NxC"""
    l = pc.shape[0]
    centriod = np.mean(pc, axis=0)
    pc = pc - centriod
    furthest_distance = np.max(np.sqrt(np.sum(pc**2, axis=1))) # 每一行(每个点的坐标)的值进行处理
    pc = pc/furthest_distance
    return pc


# def pc_normalize(pc):
#     l = pc.shape[0]
#     min = np.min(pc, axis=0)
#     max = np.max(pc, axis=0)
#     scaled_unit = 1.0 / (max - min)
#     pc = pc * scaled_unit - min * scaled_unit
#     return pc


def pc_to_df(pc):
    points = pc.points
    return pa_to_df(points)


def load_pc(path):
    logger.debug(f"Loading PC {path}")
    pc = PyntCloud.from_file(path)
    ret = df_to_pc(pc.points)
    logger.debug(f"Loaded PC {path}")

    return ret


def write_pc(path, pc):
    df = pc_to_df(pc)
    write_df(path, df)


def write_df(path, df):
    pc = PyntCloud(df)
    pc.to_file(path)


def get_files(input_glob):
    return np.array(glob(input_glob, recursive=True))


def load_points_func(x):
    return load_pc(x).points


def load_points(files, batch_size=32):
    files_len = len(files)
    with multiprocessing.Pool() as p:
        logger.info('Loading PCs into memory (parallel reading)')
        f = functools.partial(load_points_func)
        points = np.array(list(tqdm(p.imap(f, files, batch_size), total=files_len)))
    return points


