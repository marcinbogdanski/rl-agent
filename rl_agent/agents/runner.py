


import matplotlib.pyplot as plt
import time

from .agent import Agent

import pdb

def test_run(env, agent,

    nb_episodes, nb_total_steps, expl_start,

    agent_nb_actions,

    plotter=None,
    logger=None, 
    timing_arr=None, timing_dict=None, 
    seed=None):

    

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
        
        agent.reset()

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

            if plotter is not None and total_step >= agent.nb_rand_steps:
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
                if plotter is not None: # and total_step >= agent.nb_rand_steps:
                    plotter.process(logger, total_step)
                timing_dict['main_plot'] += time.time() - time_start

                time_start = time.time()
                agent.advance_one_step()
                timing_dict['main_agent_advance_step'] += time.time() - time_start
                break

        timing_dict['total'] += time.time() - time_total_start

    return agent