import numpy as np
import unittest

import rl_agent as rl

class TestPolicyGradient(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(0)

    def test_swap_actions(self):
        
        # Sutton Barto example 13.1
        env = rl.envs.SwapActions()

        agent = rl.AgentActorCritic(
            state_space=env.observation_space,
            action_space=env.action_space,
            discount=1.0,
            algorithm_type='mc_return',
            algorithm_mode='batch',
            algorithm_ep_in_batch=1,
            memory=rl.MemoryBasic(10000),
            v_fun_approx=None,
            q_fun_approx=None,
            policy=rl.PolicyTabularCat(
                learn_rate=0.00001)
            )

        done = True
        for i in range(10019):
            if done:
                obs, rew, done = env.reset(), None, False
            else:
                obs, rew, done = env.step(act)
            agent.observe(obs, rew, done)
            agent.learn()
            act = agent.take_action(obs)
            agent.next_step(done)

        print('{0[0]} {0[1]}'.format(agent.policy._weights[0]))

        self.assertTrue(agent.policy._weights[0,0] == -0.024334308254129137)
        self.assertTrue(agent.policy._weights[0,1] == 0.024334308254129144)
