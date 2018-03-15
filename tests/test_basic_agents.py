import unittest
import os
import random
import numpy as np
import tensorflow as tf

import gym

import rl_agent as rl

class TestAgent(unittest.TestCase):
    """Couple unit tests for:
        * Agent with memory-replay and DQN implemented in keras
        * Simple agent with tiles based approximator
        * Simple agent with aggregate approximator
    """


    def setUp(self):
        self.seed = 1
        rl.util.try_freeze_random_seeds(self.seed, True)

        
    def tearDown(self):
        pass


    def test_sarsa_tiles(self):
        def on_step_end(agent, reward, observation, done, action):
            if agent.total_step % 1000 == 0:
                print('test_sarsa_tiles', agent.total_step)
            if done:
                print('episode terminated at', agent.total_step)

        env = gym.make('MountainCar-v0').env
        env.seed(self.seed)

        agent = rl.AgentQ(
            state_space=env.observation_space,
            action_space=env.action_space,
            discount=0.99,
            q_fun_approx=rl.QFunctTiles(
                step_size=0.3,
                num_tillings=8,
                init_val=0),
            policy=rl.PolicyEpsGreedy(
                expl_start=False,
                nb_rand_steps=0,
                e_rand_start=1.0,
                e_rand_target=0.1,
                e_rand_decay=1/10000))

        agent.register_callback('on_step_end', on_step_end)

        rl.train_agent(env=env, agent=agent, total_steps=5000)

        # This is used to test for any numerical discrepancy between runs
        fp, ws, st, act, rew, done = agent.get_fingerprint()
        print('FINGERPRINT:', fp)
        print('  wegight sum:', ws)
        print('  st, act, rew, done:', st, act, rew, done)

        self.assertEqual(fp, -3667.665666738285)
        self.assertEqual(ws, -1297.1708778794816)
        self.assertEqual(st, -2430.494788858803)
        self.assertEqual(act, 5058)
        self.assertEqual(rew, -4999.0)
        self.assertEqual(done, 1)


    def test_sarsa_aggregate(self):
        def on_step_end(agent, reward, observation, done, action):
            if agent.total_step % 1000 == 0:
                print('test_sarsa_aggregate', agent.total_step)
            if done:
                print('episode terminated at', agent.total_step)

        env = gym.make('MountainCar-v0').env
        env.seed(self.seed)

        agent = rl.AgentQ(
            state_space=env.observation_space,
            action_space=env.action_space,
            discount=0.99,
            q_fun_approx=rl.QFunctAggregate(
                step_size=0.3,
                bins=[64, 64],
                init_val=0),
            policy=rl.PolicyEpsGreedy(
                expl_start=False,
                nb_rand_steps=0,
                e_rand_start=0.1,
                e_rand_target=0.1,
                e_rand_decay=1/10000))

        agent.register_callback('on_step_end', on_step_end)

        rl.train_agent(env=env, agent=agent, total_steps=30000)

        # This is used to test for any numerical discrepancy between runs
        fp, ws, st, act, rew, done = agent.get_fingerprint()
        print('FINGERPRINT:', fp)
        print('  wegight sum:', ws)
        print('  st, act, rew, done:', st, act, rew, done)

        self.assertEqual(fp, -24059.666698709698)
        self.assertEqual(ws, -8850.374069905585)
        self.assertEqual(st, -15178.292628804113)
        self.assertEqual(act, 29967)
        self.assertEqual(rew, -29999.0)
        self.assertEqual(done, 1)
