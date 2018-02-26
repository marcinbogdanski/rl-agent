import argparse

import os
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide TF info msgs, keep warnings

import subprocess
import socket
import datetime

import time
import pdb

import gym

import rl_agent as rl


# TODO: move globals into Program class
logger = None
plotter = None
seed = None

def on_step_end(agent, reward, observation, done, action):
    global plotter

    if agent.total_step % 1000 == 0:
        print()
        print('total_step', agent.total_step,
            'e_rand', agent._epsilon_random,
            'step_size', agent._step_size)
        print('EPISODE', agent.completed_episodes, agent.get_avg_reward(50))

    if plotter is not None:
        plotter.process(agent.logger, agent.total_step)
        if agent.total_step >= agent.nb_rand_steps:
            res = plotter.conditional_plot(agent.logger, agent.total_step)
            if res:
                plt.pause(0.001)
                pass

    if done:
        print('espiode finished after iteration', agent.step)

def test_single():
    
    env = gym.make('MountainCar-v0').env
    if seed is not None:
        env.seed(seed)

    agent = rl.Agent(
        nb_actions=3,
        discount=0.99,
        expl_start=False,
        nb_rand_steps=100000,
        e_rand_start=1.0,
        e_rand_target=0.1,
        e_rand_decay=1/10000,
        mem_size_max=100000,
        mem_enable_pmr=False,
        approximator='keras',
        step_size=0.00025,
        batch_size=1024,

        logger=logger,
        seed=seed)

    agent.register_callback('on_step_end', on_step_end)

    rl.train_agent(env=env, agent=agent, total_steps=200000, target_avg_reward=-200)

    print('='*80)
    
    fp, ws, st, act, rew, done = agent.get_fingerprint()
    print('FINGERPRINT:', fp)
    print('  wegight sum:', ws)
    print('  st, act, rew, done:', st, act, rew, done)




def main():
    global logger, plotter, seed

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, help='Random number generators seeds. Randomised by default.')
    parser.add_argument('-r', '--reproducible', action='store_true', help='If specified, will force execution on CPU within single thread for reproducible results')    
    parser.add_argument('--plot', action='store_true', help='Enable real time plotting')
    parser.add_argument('--logfile', type=str, help='Output log to specified file')
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
        seed = args.seed
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
    if args.logfile is not None or args.plot:
        curr_datetime = str(datetime.datetime.now())  # date and time
        hostname = socket.gethostname()  # name of PC where script is run
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
        git_hash = res.stdout.decode('utf-8')  # git revision if any
        logger = rl.logger.Logger(curr_datetime, hostname, git_hash)
    else:
        logger = None

    #
    #   Init plotter
    #
    if args.plot:
        fig = plt.figure()
        ax_qmax_wf = fig.add_subplot(2,4,1, projection='3d')
        ax_qmax_im = fig.add_subplot(2,4,2)
        ax_policy = fig.add_subplot(2,4,3)
        ax_trajectory = fig.add_subplot(2,4,4)
        ax_stats = None # fig.add_subplot(165)
        ax_memory = None # fig.add_subplot(2,1,2)
        ax_q_series = None # fig.add_subplot(155)
        ax_reward = fig.add_subplot(2,1,2)
        plotter = rl.logger.Plotter(  realtime_plotting=True,
                                      plot_every=1000,
                                      disp_len=1000,
                                      ax_qmax_wf=ax_qmax_wf,
                                      ax_qmax_im=ax_qmax_im,
                                      ax_policy=ax_policy,
                                      ax_trajectory=ax_trajectory,
                                      ax_stats=ax_stats,
                                      ax_memory=ax_memory,
                                      ax_q_series=ax_q_series,
                                      ax_reward=ax_reward  )
    else:
        plotter = None

    #
    #   Run application
    #
    try:
        test_single()
    finally:
        if args.logfile is not None:
            logger.save(args.logfile)
            print('Log saved')
    
    if plotter is not None:
        plt.show()


if __name__ == '__main__':
    main()
    
