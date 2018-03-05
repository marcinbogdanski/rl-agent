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
            print('EP', agent.completed_episodes, agent.get_avg_ep_reward(50))
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
        # .env at the end removes time limit, see:
        # https://stackoverflow.com/questions/42787924/
        # why-is-episode-done-after-200-time-steps-gym-environment-mountaincar
        self.env = gym.make('MountainCar-v0').env
            
        self.env.seed(args.seed)


        #
        #   Model
        #
        q_model = tf.keras.models.Sequential()
        q_model.add(tf.keras.layers.Dense(256, 'relu', input_dim=2))
        q_model.add(tf.keras.layers.Dense(256, 'relu'))
        q_model.add(tf.keras.layers.Dense(3, 'linear'))
        q_model.compile(loss='mse', 
            optimizer=tf.keras.optimizers.RMSprop(lr=0.00025))


        #
        #   Agent
        #
        agent = rl.Agent(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            discount=0.99,
            start_learning_at=0,
            
            mem_size_max=10000,
            mem_batch_size=64,
            mem_enable_pmr=False,
            q_fun_approx=rl.TilesApproximator(
                step_size=0.3,
                num_tillings=8,
                init_val=0),
            # q_fun_approx=rl.AggregateApproximator(
            #     step_size=0.3,
            #     bins=[64, 64],
            #     init_val=0),
            policy=rl.QMaxPolicy(
                expl_start=False,
                nb_rand_steps=0,
                e_rand_start=1.0,
                e_rand_target=0.1,
                e_rand_decay=1/10000)
            )


        #
        #   Plotting
        #
        # Need to re-think how plotting works
        if args.plot:
            fig1 = plt.figure()
            #fig2 = plt.figure()
            self.plotter = rl.util.Plotter(
                realtime_plotting=True, plot_every=1000, disp_len=1000,
                nb_actions=self.env.action_space.n,
                figures=(fig1, ),
                ax_qmax_wf=fig1.add_subplot(2,4,1, projection='3d'),
                ax_qmax_im=fig1.add_subplot(2,4,2),
                ax_policy=fig1.add_subplot(2,4,3),
                ax_trajectory=fig1.add_subplot(2,4,4),
                ax_stats=None,
                ax_memory=None, #fig2.add_subplot(1,1,1),
                ax_q_series=None,
                ax_reward=fig1.add_subplot(2,1,2),
            )
            self.plotter.set_state_action_spaces(
                self.env.observation_space.low, 
                self.env.observation_space.high, 
                h_line=0.0, v_line=-0.5)

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
            agent.memory.install_logger(self.logger.memory, log_every=1000)
            agent.Q.install_logger(
                self.logger.q_val, log_every=1000, samples=(64, 64))

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
    
