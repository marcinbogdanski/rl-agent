"""
This file contains couple of commonly used helper functions, mostly for:
 * parsing common arguments (seed, reproducible mode)
 * freezing random seeds across all possible libraries
"""
import os
import random
import numpy as np

import tensorflow as tf

import gym

import argparse

from . import logger

def parse_common_args(parser=None):
    """Parse commonly used command line arguments

    Params:
        parser - argparse.ArgumentParser() created by user.
                 If None, then will create new one.

    Returns:
        set of parsed arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    else:
        # TODO: enusure user did not create overlapping arguments so we 
        #       don't add them twice? return error if they did
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
    """Will attempt to make execution fully reproducible

    Params:
        seed (int): Set random seeds for following modules:
            random, numpy.random, tensorflow, gym.spaces
        reproducible (bool): if True, then:
            Disbale GPU by setting env. var. CUDA_VISIBLE_DEVICES to '-1'
            Disable randomised hashing by setting PYTHONHASHSEED to '0'
            Force single-threadeed execution in tensorflow
    """
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
    if reproducible:
        config = tf.ConfigProto()    
        config.intra_op_parallelism_threads=1
        config.inter_op_parallelism_threads=1
        sess = tf.Session(config=config)

