import os
import random
import numpy as np

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide TF info msgs, keep warnings

import gym

import argparse

from . import logger

def parse_common_args(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    else:
        # TODO: check seed, reproducible, plot and logfile exist
        #       and return error if they do
        parser = parser
        
    parser.add_argument('-s', '--seed', type=int,
        help='Random number generators seeds. Randomised by default.')
    parser.add_argument('-r', '--reproducible', action='store_true',
        help='Forces execution in single CPU thread for reproducible results')    
    parser.add_argument('--plot', action='store_true',
        help='Enable real time plotting')
    parser.add_argument('--logfile', type=str,
        help='Output log to specified file')
    args = parser.parse_args()

    if args.reproducible and args.seed is None:
        print('Error: --reproducible requires --seed to be specified as well.')
        exit(0)

    return args


def try_freeze_random_seeds(seed, reproducible):
    #
    #   Environment variables
    #
    if reproducible:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
        os.environ['PYTHONHASHSEED'] = '0'         # force reproducible hasing

    #
    #   Random seeds
    #
    print('Using random seed:', seed)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)
    # always call this, if not called expicitly, defaults to seed==0
    gym.spaces.seed(seed)

    #
    #   Set TF session
    #
    config = tf.ConfigProto()    
    if reproducible:
        config.intra_op_parallelism_threads=1
        config.inter_op_parallelism_threads=1

    config.gpu_options.per_process_gpu_memory_fraction=0.2
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

