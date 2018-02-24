import argparse

import os
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow as tf

import subprocess
import socket
import datetime

import time
import pdb

import gym

import rl_agent as rl




def test_single(logger, seed=None):
    
    logger.agent = rl.logger.Log('Agent')
    logger.q_val = rl.logger.Log('Q_Val')
    logger.env = rl.logger.Log('Environment', 'Mountain Car')
    logger.hist = rl.logger.Log('History', 'History of all states visited')
    logger.memory = rl.logger.Log('Memory', 'Agent full memory dump on given timestep')
    logger.approx = rl.logger.Log('Approx', 'Approximator')

    timing_arr = []
    timing_dict = {}

    plotting_enabled = True
    if plotting_enabled:
        fig = plt.figure()
        ax_qmax_wf = fig.add_subplot(2,4,1, projection='3d')
        ax_qmax_im = fig.add_subplot(2,4,2)
        ax_policy = fig.add_subplot(2,4,3)
        ax_trajectory = fig.add_subplot(2,4,4)
        ax_stats = None # fig.add_subplot(165)
        ax_memory = None # fig.add_subplot(2,1,2)
        ax_q_series = None # fig.add_subplot(155)
        ax_avg_reward = fig.add_subplot(2,1,2)
    else:
        ax_qmax_wf = None
        ax_qmax_im = None
        ax_policy = None
        ax_trajectory = None
        ax_stats = None
        ax_memory = None
        ax_q_series = None
        ax_avg_reward = None

    plotter = rl.logger.Plotter(plotting_enabled=plotting_enabled,
                      plot_every=1000,
                      disp_len=1000,
                      ax_qmax_wf=ax_qmax_wf,
                      ax_qmax_im=ax_qmax_im,
                      ax_policy=ax_policy,
                      ax_trajectory=ax_trajectory,
                      ax_stats=ax_stats,
                      ax_memory=ax_memory,
                      ax_q_series=ax_q_series,
                      ax_avg_reward=ax_avg_reward)


    approximator='aggregate'

    env = gym.make('MountainCar-v0').env
    if seed is not None:
        env.seed(seed)

    agent = rl.Agent(
        nb_actions=3,
        discount=0.99,
        expl_start=False,
        nb_rand_steps=0,
        e_rand_start=0.1,
        e_rand_target=0.1,
        e_rand_decay=1/10000,
        mem_size_max=100000,
        mem_enable_pmr=False,
        approximator=approximator,
        step_size=0.3,
        batch_size=1024,

        log_agent=logger.agent,
        log_q_val=logger.q_val,
        log_hist=logger.hist,
        log_memory=logger.memory,
        log_approx=logger.approx,
        seed=seed)

    rl.test_run(
        env=env,
        agent=agent,

        nb_episodes=None,
        nb_total_steps=25000,
        expl_start=False,

        agent_nb_actions=3,
        
        plotter=plotter,
        logger=logger,
        timing_arr=timing_arr,
        timing_dict=timing_dict,
        seed=seed)

    print()
    print(str.upper(approximator))
    for key in timing_arr:
        print(key, round(timing_dict[key], 3))

    fp, ws, st, act, rew, done = agent.get_fingerprint()
    print('FINGERPRINT:', fp)
    print('  wegight sum:', ws)
    print('  st, act, rew, done:', st, act, rew, done)

    if plotting_enabled:
        plt.show()



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, help='Random number generators seeds. Randomised by default.')
    parser.add_argument('-r', '--reproducible', action='store_true', help='If specified, will force execution on CPU within single thread for reproducible results')    
    args = parser.parse_args()

    if args.reproducible and args.seed is None:
        print('Error: --reproducible requires --seed to be specified as well.')
        exit(0)

    #
    #   Environment variables
    #
    if args.reproducible:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
        os.environ['PYTHONHASHSEED'] = '0'         # force reproducible hasing

    #
    #   Random seeds
    #
    if args.seed is not None:
        print('Using random seed:', args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)

    #
    #   Set TF session
    #
    config = tf.ConfigProto()    
    if args.reproducible:
        config.intra_op_parallelism_threads=1
        config.inter_op_parallelism_threads=1

    config.gpu_options.per_process_gpu_memory_fraction=0.2
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    #
    #   Init logger
    #
    curr_datetime = str(datetime.datetime.now())  # date and time
    hostname = socket.gethostname()  # name of PC where script is run
    res = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_hash = res.stdout.decode('utf-8')  # git revision if any
    logger = rl.logger.Logger(curr_datetime, hostname, git_hash)

    #
    #   Run application
    #
    try:
        test_single(logger, args.seed)
    finally:
        logger.save('data.log')
        print('log saved')


if __name__ == '__main__':
    main()
    
