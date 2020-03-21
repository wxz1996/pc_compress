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
import gzip
from tqdm import tqdm
import data_provider
import open3d as o3d
import glob
import fpzip

np.random.seed(42)
tf.set_random_seed(42)

################################################################################
### Script
################################################################################
TYPE = np.uint16
DTYPE = np.dtype(np.uint16)
SHAPE_LEN = 3
def compress(nn_output):
    string = nn_output['string']
    str_len = len(string)
    byte_len = np.array(str_len, dtype=TYPE).tobytes()
    representation = byte_len + string
    return representation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compress.py',
        description='Compress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',
        default='../ply_data',
        help='Input directory.')
    parser.add_argument(
        '--input_pattern',
        default='*.ply',
        help='Mesh detection pattern.')
    parser.add_argument(
        '--source_extension',
        help='Mesh files extension',
        default='.ply')
    parser.add_argument(
        '--output_dir',
        default='../output',
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
        '--knn',
        type=int, help='k-nearest neighbors.', default=12)
    parser.add_argument(
        '--read_batch_size', type=int, default=1,
        help='Batch size for parallel reading.')
    parser.add_argument(
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')

    args = parser.parse_args()

    assert args.batch_size > 0, 'batch_size must be positive'


    args.input_dir = os.path.normpath(args.input_dir)
    len_input_dir = len(args.input_dir)
    assert os.path.exists(args.input_dir), "Input directory not found"

    input_glob = os.path.join(args.input_dir, args.input_pattern)
    files = pc_io.get_files(input_glob)
    assert len(files) > 0, "No input files found"
    filenames = [x[len_input_dir+1:] for x in files]
    output_files = [os.path.join(args.output_dir, x + '.bin') for x in filenames]
    output_infs = [os.path.join(args.output_dir, x + '.inf.bin') for x in filenames]
    pcQueue_512, pcQueue_1024, pcQueue_2048 = data_provider.load_data(
        args.input_dir, args.source_extension)

    centroid_list = []
    furthest_distance_list = []
    meta_matrix, nor_pc_512, nor_pc_1024, nor_pc_2048 = data_provider.gen_meta(centroid_list,furthest_distance_list,pcQueue_512,pcQueue_1024,pcQueue_2048)
    compressed_bytes = fpzip.compress(meta_matrix.astype(np.float32), precision=0, order='C')
    with open(output_infs[0], 'wb') as f:
        f.write(compressed_bytes)
    f.close()

    def patch_compress(points,model,checkpoint):
        estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=checkpoint,
            params={
                'checkpoint_dir': checkpoint,
                'batch_size': args.batch_size,
                'knn': args.knn,
            })
        result = estimator.predict(
                input_fn=lambda: model.input_fn(points, args.batch_size, args.preprocess_threads, repeat=False, is_train = False),
                predict_keys=['string', 'x_tilde', 'y_tilde'])
        return result

    output_list = []
    output_list.append(patch_compress(nor_pc_512, compression_model_512, args.checkpoint_dir_512))
    output_list.append(patch_compress(nor_pc_1024, compression_model_1024, args.checkpoint_dir_1024))
    output_list.append(patch_compress(nor_pc_2048, compression_model_2048, args.checkpoint_dir_2048))
    with gzip.open(output_files[0], "ab") as f:
        patch_num_512 = len(nor_pc_512)
        patch_num_1024 = len(nor_pc_1024)
        patch_num_2048 = len(nor_pc_2048)
        patch_512_byte = np.array(patch_num_512, dtype=TYPE).tobytes()
        patch_1024_byte = np.array(patch_num_1024, dtype=TYPE).tobytes()
        patch_2048_byte = np.array(patch_num_2048, dtype=TYPE).tobytes()
        f.write(patch_512_byte + patch_1024_byte + patch_2048_byte)
        for i in range(len(output_list)):
            for ret in output_list[i]:
                representation = compress(ret)
                f.write(representation)
