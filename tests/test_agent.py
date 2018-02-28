import unittest
import os
import random
import numpy as np
import tensorflow as tf

import gym

import rl_agent as rl

class TestAgent(unittest.TestCase):

    def setUp(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU
        os.environ['PYTHONHASHSEED'] = '0'         # force reproducible hasing

        #
        #   Random seeds
        #
        self.seed = 1
        print('Using random seed:', self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        #
        #   TF session
        #
        config = tf.ConfigProto()    
        config.intra_op_parallelism_threads=1
        config.inter_op_parallelism_threads=1

        config.gpu_options.per_process_gpu_memory_fraction=0.2
        self.sess = tf.Session(config=config)

        #
        #   Logger etc
        #
        self.logger = rl.logger.Logger('curr_datetime', 'hostname', 'git_hash')
        self.logger.agent = rl.logger.Log('Agent')
        self.logger.q_val = rl.logger.Log('Q_Val')
        self.logger.env = rl.logger.Log('Environment', 'Mountain Car')
        self.logger.hist = rl.logger.Log('History', 'History of all states visited')
        self.logger.memory = rl.logger.Log('Memory', 'Agent full memory dump on given timestep')
        self.logger.approx = rl.logger.Log('Approx', 'Approximator')

        
    def tearDown(self):
        pass

    def test_10_run_keras_1(self):
        def on_step_end(agent, reward, observation, done, action):
            if agent.total_step % 1000 == 0:
                print('test_10_run_keras_1', agent.total_step)
            if done:
                print('episode terminated at', agent.total_step)

        env = gym.make('MountainCar-v0').env
        env.seed(self.seed)

        q_model = tf.keras.models.Sequential()
        q_model.add(tf.keras.layers.Dense(units=256, activation='relu', input_dim=2))
        q_model.add(tf.keras.layers.Dense(units=256, activation='relu'))
        q_model.add(tf.keras.layers.Dense(units=3, activation='linear'))
        q_model.compile(loss='mse', 
            optimizer=tf.keras.optimizers.RMSprop(lr=0.00025))

        agent = rl.Agent(
            state_space=env.observation_space,
            action_space=env.action_space,
            discount=0.99,
            expl_start=False,
            nb_rand_steps=22000,
            e_rand_start=1.0,
            e_rand_target=0.1,
            e_rand_decay=1/10000,
            mem_size_max=10000,
            mem_enable_pmr=False,
            q_fun_approx=rl.KerasApproximator(
                discount=0.99,
                model=q_model),
            step_size=0.00025,
            batch_size=64,

            logger=None)

        agent.register_callback('on_step_end', on_step_end)

        rl.train_agent(env=env, agent=agent, total_steps=23000)

        fp, ws, st, act, rew, done = agent.get_fingerprint()
        print('FINGERPRINT:', fp)
        print('  wegight sum:', ws)
        print('  st, act, rew, done:', st, act, rew, done)

        self.assertEqual(fp, -8093.627516248174)
        self.assertEqual(ws, 3562.3154466748238)
        self.assertEqual(st, -11822.942962922998)
        self.assertEqual(act, 23165)
        self.assertEqual(rew, -22999.0)
        self.assertEqual(done, 1)

    def test_20_run_tile_1(self):
        def on_step_end(agent, reward, observation, done, action):
            if agent.total_step % 1000 == 0:
                print('test_20_run_tile_1', agent.total_step)
            if done:
                print('episode terminated at', agent.total_step)

        env = gym.make('MountainCar-v0').env
        env.seed(self.seed)

        agent = rl.Agent(
            state_space=env.observation_space,
            action_space=env.action_space,
            discount=0.99,
            expl_start=False,
            nb_rand_steps=0,
            e_rand_start=1.0,
            e_rand_target=0.1,
            e_rand_decay=1/10000,
            mem_size_max=10000,
            mem_enable_pmr=False,
            q_fun_approx='tiles',
            step_size=0.3,
            batch_size=64,

            logger=None)

        agent.register_callback('on_step_end', on_step_end)

        rl.train_agent(env=env, agent=agent, total_steps=5000)

        fp, ws, st, act, rew, done = agent.get_fingerprint()
        print('FINGERPRINT:', fp)
        print('  wegight sum:', ws)
        print('  st, act, rew, done:', st, act, rew, done)

        self.assertEqual(fp, -3695.0214844234456)
        self.assertEqual(ws, -1310.189102073466)
        self.assertEqual(st, -2456.8323823499795)
        self.assertEqual(act, 5068)
        self.assertEqual(rew, -4998.0)
        self.assertEqual(done, 2)


    def test_30_run_aggregate_1(self):
        def on_step_end(agent, reward, observation, done, action):
            if agent.total_step % 1000 == 0:
                print('test_30_run_aggregate_1', agent.total_step)
            if done:
                print('episode terminated at', agent.total_step)

        env = gym.make('MountainCar-v0').env
        env.seed(self.seed)

        agent = rl.Agent(
            state_space=env.observation_space,
            action_space=env.action_space,
            discount=0.99,
            expl_start=False,
            nb_rand_steps=0,
            e_rand_start=0.1,
            e_rand_target=0.1,
            e_rand_decay=1/10000,
            mem_size_max=10000,
            mem_enable_pmr=False,
            q_fun_approx='aggregate',
            step_size=0.3,
            batch_size=64,

            logger=None)

        agent.register_callback('on_step_end', on_step_end)

        rl.train_agent(env=env, agent=agent, total_steps=30000)

        fp, ws, st, act, rew, done = agent.get_fingerprint()
        print('FINGERPRINT:', fp)
        print('  wegight sum:', ws)
        print('  st, act, rew, done:', st, act, rew, done)

        self.assertEqual(fp, -24664.755330808075)
        self.assertEqual(ws, -9677.655505872985)
        self.assertEqual(st, -15033.09982493509)
        self.assertEqual(act, 30044)
        self.assertEqual(rew, -29999.0)
        self.assertEqual(done, 1)
