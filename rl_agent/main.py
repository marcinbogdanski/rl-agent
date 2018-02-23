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
from .mountain_car import MountainCarEnv

from .agent import Agent

from .logger import Logger, Log
from .log_viewer import Plotter


def test_run(nb_episodes, nb_total_steps, expl_start,

    agent_nb_actions,
    agent_discount,
    agent_nb_rand_steps,
    agent_e_rand_start,
    agent_e_rand_target,
    agent_e_rand_decay,

    mem_size_max,
    mem_enable_pmr,

    approximator, step_size, batch_size,
    plotter=None,
    logger=None, 
    timing_arr=None, timing_dict=None, 
    seed=None):

    env = gym.make('MountainCar-v0').env
    if seed is not None:
        env.seed(seed)
    #env = MountainCarEnv(log=logger.env)

    agent = Agent(nb_actions=agent_nb_actions,
                discount=agent_discount,
                nb_rand_steps=agent_nb_rand_steps,
                e_rand_start=agent_e_rand_start,
                e_rand_target=agent_e_rand_target,
                e_rand_decay=agent_e_rand_decay,
                mem_size_max=mem_size_max,
                mem_enable_pmr=mem_enable_pmr,
                approximator=approximator,
                step_size=step_size,
                batch_size=batch_size,
                log_agent=logger.agent,
                log_q_val=logger.q_val,
                log_hist=logger.hist,
                log_memory=logger.memory,
                log_approx=logger.approx,
                seed=seed)

    timing_arr.append('total')
    timing_arr.append('main_reset')
    timing_arr.append('main_agent_pick_action')
    timing_arr.append('main_agent_append_action')
    timing_arr.append('main_agent_log')
    timing_arr.append('main_plot')
    timing_arr.append('main_agent_advance_step')
    timing_arr.append('main_env_step')
    timing_arr.append('main_agent_append_trajectory')
    timing_arr.append('main_agent_td_online')
    timing_arr.append('  eval_td_start')
    timing_arr.append('  eval_td_get_batch')
    timing_arr.append('  eval_td_update')
    timing_arr.append('    update_loop')
    timing_arr.append('      update_loop_pred')
    timing_arr.append('    update_convert_numpy')
    timing_arr.append('    update_train_on_batch')
    timing_arr.append('    update2_create_arr')
    timing_arr.append('    update2_loop')
    timing_arr.append('    update2_scale')
    timing_arr.append('    update2_predict')
    timing_arr.append('    update2_post')
    timing_arr.append('    update2_train_on_batch')
    # timing_arr.append('hohohohoho')
    timing_dict.clear()
    for string in timing_arr:
        timing_dict[string] = 0

    #
    #   Initialise loggers
    #
    episode = -1
    total_step = -1
    while True:

        PRINT_FROM = 100000
        
        episode += 1
        if nb_episodes is not None and episode >= nb_episodes:
            break

        if nb_total_steps is not None and total_step >= nb_total_steps:
            break

        time_total_start = time.time()
        
        step = 0
        total_step += 1

        print('episode:', episode, '/', nb_episodes,
            'step', step, 'total_step', total_step)

        time_start = time.time()

        print('-------------------------- total step -- ', total_step)

        obs = env.reset()
        print('STEP', obs)
        # obs = env.reset(expl_start=expl_start)
        agent.reset(expl_start=expl_start)

        agent.append_trajectory(observation=obs,
                                reward=None,
                                done=None)

        timing_dict['main_reset'] += time.time() - time_start

        while True:

            # if step % 3 == 0:
            time_start = time.time()
            action = agent.pick_action(obs)
            if total_step >= PRINT_FROM:
                print('ACTION', action)
                print('MEM_LEN', agent._memory._curr_len)
                print('EPSILON', agent._epsilon_random)
            timing_dict['main_agent_pick_action'] += time.time() - time_start

            time_start = time.time()
            agent.append_action(action=action)
            timing_dict['main_agent_append_action'] += time.time() - time_start

            time_start = time.time()
            agent.log(episode, step, total_step)
            timing_dict['main_agent_log'] += time.time() - time_start

            time_start = time.time()
            if total_step % 1000 == 0:

                print()
                print('total_step', total_step,
                    'e_rand', agent._epsilon_random, 
                    'step_size', agent._step_size)

                # PRINT TIMING STATS
                # for key in timing_arr:
                #     print(key, round(timing_dict[key], 3))

                # PRINT (RAND)
                # i = total_step
                # t_steps = logger.agent.total_steps[0:i:1]
                # ser_e_rand = logger.agent.data['e_rand'][0:i:1]
                # ser_rand_act = logger.agent.data['rand_act'][0:i:1]
                # ser_mem_size = logger.agent.data['mem_size'][0:i:1]
                # arr = logger.agent.data['rand_act'][max(0, i-1000):i]
                # nz = np.count_nonzero(arr)
                # print('RAND: ', nz, ' / ', len(arr))

            if plotter is not None and total_step >= agent_nb_rand_steps:
                plotter.process(logger, total_step)
                res = plotter.conditional_plot(logger, total_step)
                if res:
                    plt.pause(0.001)
                    pass

            timing_dict['main_plot'] += time.time() - time_start
            

            time_start = time.time()
            agent.advance_one_step()
            timing_dict['main_agent_advance_step'] += time.time() - time_start


            if nb_total_steps is not None and total_step >= nb_total_steps:
                break

            #if total_step >= 111280+8:
            # if total_step >= 5:
            #    pdb.set_trace()
            #    exit(0)

            #   ---   time step rolls here   ---
            step += 1
            total_step += 1
            if total_step >= PRINT_FROM:
                print('------------------------------ total step -- ', total_step)


            time_start = time.time()
            if agent_nb_actions == 2 and action == 1:
                action_p = 2
            else:
                action_p = action
            obs, reward, done, _ = env.step(action_p)
            if total_step >= PRINT_FROM:
                print('STEP', obs)
            reward = round(reward)
            timing_dict['main_env_step'] += time.time() - time_start

            time_start = time.time()
            agent.append_trajectory(
                        observation=obs,
                        reward=reward,
                        done=done)
            timing_dict['main_agent_append_trajectory'] += time.time() - time_start

            time_start = time.time()
            agent.eval_td_online(timing_dict)
            timing_dict['main_agent_td_online'] += time.time() - time_start
            
            if done or step >= 100000:
                print('espiode finished after iteration', step)
                
                time_start = time.time()
                agent.log(episode, step, total_step)
                timing_dict['main_agent_log'] += time.time() - time_start

                time_start = time.time()
                if plotter is not None: # and total_step >= agent_nb_rand_steps:
                    plotter.process(logger, total_step)
                timing_dict['main_plot'] += time.time() - time_start

                time_start = time.time()
                agent.advance_one_step()
                timing_dict['main_agent_advance_step'] += time.time() - time_start
                break

        timing_dict['total'] += time.time() - time_total_start

    return agent



def test_single(logger, seed=None):
    
    logger.agent = Log('Agent')
    logger.q_val = Log('Q_Val')
    logger.env = Log('Environment', 'Mountain Car')
    logger.hist = Log('History', 'History of all states visited')
    logger.memory = Log('Memory', 'Agent full memory dump on given timestep')
    logger.approx = Log('Approx', 'Approximator')

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

    plotter = Plotter(plotting_enabled=plotting_enabled,
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

    trained_agent = test_run(
            nb_episodes=None,
            nb_total_steps=25000,
            expl_start=False,

            agent_nb_actions=3,
            agent_discount=0.99,
            agent_nb_rand_steps=0,
            agent_e_rand_start=0.1,
            agent_e_rand_target=0.1,
            agent_e_rand_decay=1/10000,

            mem_size_max=100000,
            mem_enable_pmr=False,

            approximator=approximator,
            step_size=0.3,
            batch_size=1024,
            
            plotter=plotter,
            logger=logger,
            timing_arr=timing_arr,
            timing_dict=timing_dict,
            seed=seed)

    print()
    print(str.upper(approximator))
    for key in timing_arr:
        print(key, round(timing_dict[key], 3))

    fp, ws, st, act, rew, done = trained_agent.get_fingerprint()
    print('FINGERPRINT:', fp)
    print('  wegight sum:', ws)
    print('  st, act, rew, done:', st, act, rew, done)

    if plotting_enabled:
        plt.show()





def plot_history_3d(ax, hpos, hvel, hact, htar):
    hpos = np.array(hpos)
    hvel = np.array(hvel)
    hact = np.array(hact)
    htar = np.array(htar)

    idx = hact==1

    r_pos = hpos[idx]
    r_vel = hvel[idx]
    r_tar = htar[idx]
    

    # ax.scatter(hpos[::10], hvel[::10], -htarget[::10])
    ax.scatter(r_pos, r_vel, r_tar)


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
    logger = Logger(curr_datetime, hostname, git_hash)

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
    
