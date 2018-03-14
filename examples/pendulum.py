import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import gym

import rl_agent as rl

import tensorflow as tf


class Program():
    def __init__(self):
        self.env = None
        self.logger = None
        self.plotter = None

    def on_step_end(self, agent, reward, observation, done, action):
        """This is callback executed at step end.action

        Example:
            # time step t begins
            obs, reward, done = env.step(previous_action)
            action = agent.take_action(obs)
            on_step_end(reward, observation, done, action)
            # time step t ends
        """

        self.env.render()

        # Print to console
        if agent.total_step % 1000 == 0:
            print()
            print('total_step', agent.total_step)
            print('EP', agent.completed_episodes, agent.get_cont_reward(1000))
            if done:
                print('espiode finished after iteration', agent.step)   

        # Plot stuff
        if self.plotter is not None:
            if agent.total_step >= agent.start_learning_at:
                res = self.plotter.conditional_plot(
                    self.logger, agent.total_step)

        



    def main(self):
        
        args = rl.util.parse_common_args()
        rl.util.try_freeze_random_seeds(args.seed, args.reproducible)
        
        
        
        #
        #   Environment
        #
        # Environment outputs 3-tuple: cos(ang), sin(ang), angular-velocity
        # we translate that to 2-tuple: angle [-pi, pi], ang-vel [-8.0, 8.0]
        # so we can plot 2d action space nicely
        #
        # Environment expect continous 1-tuple action representing torque
        # in range [-2.0, 2.0], but our agent outputs categorical action 0-4
        # so we need to tranlate that to torque
        # this is becouse continous actions are not implemented yet 
        def obs_trans(obs):
            """Translate from 3d obs space to 2d (for easier plotting)"""
            theta = np.arctan2(obs[1], obs[0])
            vel = obs[2]
            return np.array([theta, vel])

        def act_trans(act):
            """Translate from categorical actions to continous"""
            torques = [-2.0, -0.5, 0.0, 0.5, 2.0]
            return np.array([torques[act]])

        self.env = rl.util.EnvTranslator(
            env=gym.make('Pendulum-v0'),
            observation_space=gym.spaces.Box(
                low=np.array([-np.pi, -8.0]), 
                high=np.array([np.pi, 8.0])),
            observation_translator=obs_trans,
            action_space=gym.spaces.Discrete(5),
            action_translator=act_trans,
            reward_translator=None)

        self.env.seed(args.seed)



        #
        #   Agent
        #
        agent = rl.AgentSARSA(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            discount=0.99,
            start_learning_at=0,
            q_fun_approx=rl.QFunctTiles(
                step_size=0.3,
                num_tillings=16,
                init_val=0),
            policy=rl.QMaxPolicy(
                expl_start=False,
                nb_rand_steps=0,
                e_rand_start=0.0,
                e_rand_target=0.0,
                e_rand_decay=1/10000)
            )


        #
        #   Plotting
        #
        # Need to re-think how plotting works
        if args.plot:
            fig1 = plt.figure()
            self.plotter = rl.util.Plotter(
                realtime_plotting=True,
                plot_every=1000,
                disp_len=1000,
                nb_actions=self.env.action_space.n,
                figures=(fig1, ),
                ax_qmax_wf=fig1.add_subplot(2,4,1, projection='3d'),
                ax_qmax_im=fig1.add_subplot(2,4,2),
                ax_policy=fig1.add_subplot(2,4,3),
                ax_trajectory=fig1.add_subplot(2,4,4),
                ax_stats=None,
                ax_memory=None,
                ax_q_series=None,
                ax_reward=fig1.add_subplot(2,1,2),
            )
            self.plotter.set_state_action_spaces(
                self.env.observation_space.low, 
                self.env.observation_space.high, 
                h_line=0.0, v_line=0.0)

        #
        #   Logging
        #
        if args.logfile is not None or args.plot:
            self.logger = rl.util.Logger()

            self.logger.agent = rl.util.Log('Agent')
            self.logger.q_val = rl.util.Log('Q_Val')
            self.logger.env = rl.util.Log('Environment')
            self.logger.hist = rl.util.Log('History', 'All sates visited')
            self.logger.memory = rl.util.Log('Memory', 'Full memory dump')
            self.logger.approx = rl.util.Log('Approx', 'Approximator')
            self.logger.epsumm = rl.util.Log('Episodes')

            agent.log_episodes = self.logger.epsumm
            agent.log_hist = self.logger.hist
            agent.Q.install_logger(
                self.logger.q_val, log_every=1000, samples=(64, 64))

        #
        #   Callback
        #
        agent.register_callback('on_step_end', self.on_step_end)

        
        #
        #   Runner
        #
        try:
            rl.train_agent(
                env=self.env,
                agent=agent, 
                total_steps=1000000,
                target_avg_reward=-200)
        finally:
            if args.logfile is not None:
                logger.save(args.logfile)
                print('Log saved')
        
        if self.plotter is not None:
            plt.show()


if __name__ == '__main__':
    prog = Program()
    prog.main()
    
