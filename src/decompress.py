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
import numpy as np
import tensorflow as tf
import argparse
import compression_model_512
import compression_model_1024
import compression_model_2048
import pc_io
import multiprocessing
import gzip
import fpzip
from tqdm import tqdm

np.random.seed(42)
tf.set_random_seed(42)

# Use CPU
# For unknown reasons, this is 3 times faster than GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

################################################################################
### Script
################################################################################
TYPE = np.uint16
DTYPE = np.dtype(TYPE)
SHAPE_LEN = 3


def load_compressed_file(file):
    strings_512 = []
    string_1024 = []
    string_2048 = []
    with gzip.open(file, "rb") as f:
        nums_512 = np.frombuffer(f.read(DTYPE.itemsize), dtype=TYPE)
        nums_1024 = np.frombuffer(f.read(DTYPE.itemsize), dtype=TYPE)
        nums_2048 = np.frombuffer(f.read(DTYPE.itemsize), dtype=TYPE)
        for i in range(nums_512[0]):
            str_len = np.frombuffer(f.read(DTYPE.itemsize), dtype=TYPE)
            string = f.read(str_len[0])
            strings_512.append(string)
        for i in range(nums_1024[0]):
            str_len = np.frombuffer(f.read(DTYPE.itemsize), dtype=TYPE)
            string = f.read(str_len[0])
            string_1024.append(string)
        for i in range(nums_2048[0]):
            str_len = np.frombuffer(f.read(DTYPE.itemsize), dtype=TYPE)
            string = f.read(str_len[0])
            string_2048.append(string)
        return strings_512, string_1024, string_2048


def load_compressed_files(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading data into memory (parallel reading)')
        data = np.array(list(tqdm(p.imap(load_compressed_file, files, batch_size), total=files_len)))

    return data


def load_compressed_inf(file):
    COR_SIZE = 4
    FUR_SIZE = 4
    POS = 0
    with open(file, 'rb') as f:
        compressed_bytes = f.read()
        data_again = fpzip.decompress(compressed_bytes, order='C')
        data_again = np.squeeze(data_again)
    return data_again


def load_compressed_infs(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        logger.info('Loading data into memory (parallel reading)')
        data = np.array(list(tqdm(p.imap(load_compressed_inf, files, batch_size), total=files_len)))

    return data


def input_fn(features, batch_size):
    with tf.device('/cpu:0'):
        zero = tf.constant(0)
        dataset = tf.data.Dataset.from_generator(lambda: features, (tf.string))
        dataset = dataset.map(lambda t: (t, zero))
        dataset = dataset.batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='decompress.py',
        description='Decompress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input_dir',
        default='../output',
        help='Input directory.')
    parser.add_argument(
        '--input_pattern',
        default='*.ply.bin',
        help='Mesh detection pattern.')
    parser.add_argument(
        '--inf_pattern',
        default='*.ply.inf.bin',
        help='inf detection pattern.')
    parser.add_argument(
        '--output_dir',
        default='../out_ply',
        help='Output directory.')
    parser.add_argument(
        '--checkpoint_dir_512',
        default='/data/wenxuanzheng/mix_compress/512/00005',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--checkpoint_dir_1024',
        default='/data/wenxuanzheng/mix_compress/1024/00005',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--checkpoint_dir_2048',
        default='/data/wenxuanzheng/mix_compress/2048/00005',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size.')
    parser.add_argument(
        '--read_batch_size', type=int, default=1,
        help='Batch size for parallel reading.')
    parser.add_argument(
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')
    parser.add_argument(
        '--output_extension', default='.ply',
        help='Output extension.')

    args = parser.parse_args()

    assert args.batch_size > 0, 'batch_size must be positive'


    args.input_dir = os.path.normpath(args.input_dir)
    len_input_dir = len(args.input_dir)
    assert os.path.exists(args.input_dir), "Input directory not found"

    input_glob = os.path.join(args.input_dir, args.input_pattern)
    inf_glob = os.path.join(args.input_dir, args.inf_pattern)
    files = pc_io.get_files(input_glob)
    infs = pc_io.get_files(inf_glob)

    assert len(files) > 0, "No input files found"
    filenames = [x[len_input_dir + 1:] for x in files]
    output_files = [os.path.join(args.output_dir, x + '.ply') for x in filenames]

    compressed_data = load_compressed_files(files, args.read_batch_size)
    compressed_infs = load_compressed_infs(infs, args.read_batch_size).reshape(-1,4)
    strings_512 = iter(compressed_data[0][0])
    strings_1024 = iter(compressed_data[0][1])
    strings_2048 = iter(compressed_data[0][2])
    def patch_decompress(compressed_strings, model, checkpoint_dir):
        estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=checkpoint_dir,
            params={
                'checkpoint_dir': checkpoint_dir,
                'decompress': True,
                'batch_size': args.batch_size
            })

        result = estimator.predict(
            input_fn=lambda: input_fn(compressed_strings, args.batch_size),
            predict_keys=['y_tilde','x_hat'])

        return result


    rec_pc = []

    rec_pc.append(patch_decompress(strings_2048, compression_model_2048, args.checkpoint_dir_2048))
    rec_pc.append(patch_decompress(strings_1024, compression_model_1024, args.checkpoint_dir_1024))
    rec_pc.append(patch_decompress(strings_512, compression_model_512, args.checkpoint_dir_512))
    pc_list = []
    for i in range(len(rec_pc)):
        for ret in rec_pc[i]:
            pc_list.append(ret['x_hat'])

    assert compressed_infs.shape[0] == len(pc_list)
    output_pc = []
    for i in range(len(pc_list)):
        centriod = compressed_infs[i][:3]
        furthest_distance = compressed_infs[i][3]
        out_pc = centriod + pc_list[i] * furthest_distance
        output_pc.append(out_pc)
    out_pc = np.vstack(output_pc[:])
    len_files = len(files)
    logger.info(f'{i}/{len_files} - Writing {files[0]} to {output_files[0]}')
    output_dir, _ = os.path.split(output_files[0])
    os.makedirs(output_dir, exist_ok=True)
    pc_io.write_df(output_files[0], pc_io.pa_to_df(out_pc))


