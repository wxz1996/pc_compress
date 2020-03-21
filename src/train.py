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
import tensorflow.compat.v1 as tf
import argparse
import compression_model_256
import compression_model_512
import compression_model_1024
import compression_model_2048
import pc_io
from tensorflow.python import debug as tf_debug
import part_dataset

np.seterr(divide='ignore',invalid='ignore')
np.random.seed(42) 
tf.set_random_seed(42)

################################################################################
### Training
################################################################################
def train():
    """Trains the model."""
    if args.verbose:
        tf.logging.set_verbosity(tf.logging.INFO) 

    files = pc_io.get_files(args.train_glob)  
    points = pc_io.load_points(files) 
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in files])
    for cat in files_cat:
        assert (cat == 'train') or (cat == 'eval')
    TRAIN_DATASET = points[files_cat == 'train']
    EVAL_DATASET = points[files_cat == 'eval']
    assert (len(TRAIN_DATASET) + len(EVAL_DATASET) == len(points))

    
    config = tf.estimator.RunConfig(
        keep_checkpoint_every_n_hours=1,
        save_checkpoints_secs=args.save_checkpoints_secs, # 600
        keep_checkpoint_max=args.keep_checkpoint_max, # 50
        log_step_count_steps=args.log_step_count_steps,  # 100
        save_summary_steps=args.save_summary_steps, # 100
        tf_random_seed=42)

    estimator = tf.estimator.Estimator(
        model_fn=compression_model_2048.model_fn, 
        model_dir=args.checkpoint_dir,  
        config=config,
        params={
            'num_points': args.num_point,
            'batch_size': args.batch_size,
            'knn': args.knn,
            'alpha': args.alpha,
            'gamma': args.gamma,
            'lmbda': args.lmbda,
            'additional_metrics': not args.no_additional_metrics,
            'checkpoint_dir': args.checkpoint_dir,
            'data_format': DATA_FORMAT # channels_first
        })

    hooks = None
    if args.debug_address is not None:
        hooks = [tf_debug.TensorBoardDebugHook(args.debug_address)]


    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: compression_model_2048.input_fn(
            TRAIN_DATASET, args.batch_size, args.preprocess_threads, prefetch_size=args.prefetch_size),
        max_steps=args.max_steps,
        hooks=hooks)

    val_spec = tf.estimator.EvalSpec(
        input_fn=lambda: compression_model_2048.input_fn(
            EVAL_DATASET, args.batch_size, args.preprocess_threads, repeat=False, prefetch_size=args.prefetch_size),
        steps=None,
        hooks=hooks)
    # 
    tf.estimator.train_and_evaluate(estimator, train_spec, val_spec)

################################################################################
### Script
################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='train.py',
        description='Train network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--train_glob',
        default='/data/wenxuanzheng/shapenet_patch_part/2048/**/*.ply',
        help='Glob pattern for identifying training data.')
    parser.add_argument(
        '--checkpoint_dir',
        default='/data/wenxuanzheng/mix_compress/2048/00005',
        help='Directory where to save/load model checkpoints.')
    parser.add_argument(
        '--num_point', type=int, default=2048,
        help='Point Number [default: 2048]')
    parser.add_argument(
        '--knn',
        type=int, help='k-nearest neighbors.', default=8)
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Report bitrate and distortion when training.')
    parser.add_argument(
        '--no_additional_metrics', action='store_true',
        help='Report additional metrics when training.')
    parser.add_argument(
        '--save_checkpoints_secs', type=int, default=600,
        help='Save checkpoints every n seconds during training.')
    parser.add_argument(
        '--keep_checkpoint_max', type=int, default=50,
        help='Maximum number of checkpoint files to keep.')
    parser.add_argument(
        '--log_step_count_steps', type=int, default=100,
        help='Log global step and loss every n steps.')
    parser.add_argument(
        '--save_summary_steps', type=int, default=100,
        help='Save summaries every n steps.')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for training.')
    parser.add_argument(
        '--prefetch_size', type=int, default=128,
        help='Number of batches to prefetch for training.')
    parser.add_argument(
        '--lmbda', type=float, default=0.0005,
        help='Lambda for rate-distortion tradeoff.')
    parser.add_argument(
        '--max_steps', type=int, default=1000000,
        help='Train up to this number of steps.')
    parser.add_argument(
        '--preprocess_threads', type=int, default=16,
        help='Number of CPU threads to use for parallel decoding.')
    parser.add_argument(
        '--debug_address', default=None,
        help='TensorBoard debug address.')


    args = parser.parse_args()

    os.makedirs(os.path.split(args.checkpoint_dir)[0], exist_ok=True)
    assert args.batch_size > 0, 'batch_size must be positive'

    train()
