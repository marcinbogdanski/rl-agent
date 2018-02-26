import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import collections
import argparse

from .logger import Logger, Log

import pdb

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Data log filename')
    args = parser.parse_args()

    logger = Logger()
    logger.load(args.filename)

    print()
    print(logger)
    print(logger.env)
    print(logger.agent)
    print(logger.q_val)
    print(logger.hist)
    print(logger.approx)

    fig = plt.figure()
    ax_qmax_wf = fig.add_subplot(151, projection='3d')
    ax_qmax_im = fig.add_subplot(152)
    ax_policy = fig.add_subplot(153)
    ax_trajectory = fig.add_subplot(154)
    ax_stats = None # fig.add_subplot(165)
    ax_memory = fig.add_subplot(155)
    ax_q_series = None # fig.add_subplot(155)


    plotter = Plotter(realtime_plotting=True,
                      plot_every=1000,
                      disp_len=1000,
                      ax_qmax_wf=ax_qmax_wf,
                      ax_qmax_im=ax_qmax_im,
                      ax_policy=ax_policy,
                      ax_trajectory=ax_trajectory,
                      ax_stats=ax_stats,
                      ax_memory=ax_memory,
                      ax_q_series=ax_q_series,
                      ax_reward=None)

    for total_step in range(0, len(logger.hist.total_steps)):
        print(total_step)

        plotter.process(logger, total_step)
        res = plotter.conditional_plot(logger, total_step)
        if res:
            plt.pause(0.1)

    plt.show()

class Plotter():
    def __init__(self, realtime_plotting, plot_every, disp_len, 
        ax_qmax_wf, ax_qmax_im, ax_policy,
        ax_trajectory, ax_stats, ax_memory, ax_q_series, ax_reward):

        self.realtime_plotting = realtime_plotting
        self.plot_every = plot_every
        self.disp_len = disp_len

        self.ax_qmax_wf = ax_qmax_wf
        self.ax_qmax_im = ax_qmax_im
        self.ax_policy = ax_policy
        self.ax_trajectory = ax_trajectory
        self.ax_stats = ax_stats
        self.ax_memory = ax_memory
        self.ax_q_series = ax_q_series
        self.ax_reward = ax_reward

        self.q_val = None
        self.ser_X =  []
        self.ser_E0 = []
        self.ser_E1 = []
        self.ser_E2 = []
        

    def process(self, logger, current_total_step):
        """Call this every step to track data"""

        if logger.q_val.data['q_val'][current_total_step] is not None:
            self.q_val = logger.q_val.data['q_val'][current_total_step]


        if logger.q_val.data['series_E0'][current_total_step] is not None:
            self.ser_X.append(current_total_step)
            self.ser_E0.append(logger.q_val.data['series_E0'][current_total_step])
            self.ser_E1.append(logger.q_val.data['series_E1'][current_total_step])
            self.ser_E2.append(logger.q_val.data['series_E2'][current_total_step])

        if logger.memory.data['hist_St'][current_total_step] is not None:
            self.hist_St = logger.memory.data['hist_St'][current_total_step]
            self.hist_At = logger.memory.data['hist_At'][current_total_step]
            self.hist_Rt_1 = logger.memory.data['hist_Rt_1'][current_total_step]
            self.hist_St_1 = logger.memory.data['hist_St_1'][current_total_step]
            self.hist_done = logger.memory.data['hist_done'][current_total_step]
            self.hist_error = logger.memory.data['hist_error'][current_total_step]

    def conditional_plot(self, logger, current_total_step):
        if current_total_step % self.plot_every == 0 and self.realtime_plotting:
            self.plot(logger, current_total_step)
            return True
        else:
            return False

    def plot(self, logger, current_total_step):
        if not self.realtime_plotting:
            return

        extent = (-1.2, 0.5, -0.07, 0.07)

        # print('---')
        # print(current_total_step % step_span == 0)
        # print(q_val is not None)

        if self.q_val is not None:
            q_max = np.max(self.q_val, axis=2)

            if self.ax_qmax_wf is not None:
                self.ax_qmax_wf.clear()
                plot_q_val_wireframe(self.ax_qmax_wf, q_max,
                    extent, ('pos', 'vel', 'q_max'))

            if self.ax_qmax_im is not None:
                self.ax_qmax_im.clear()
                plot_q_val_imshow(self.ax_qmax_im, q_max,
                    extent, h_line=0.0, v_line=-0.5)
                self.ax_qmax_im.set_yticklabels([])
                self.ax_qmax_im.set_xticklabels([])
            
            if self.ax_policy is not None:
                self.ax_policy.clear()
                plot_policy(self.ax_policy, self.q_val,
                    extent, h_line=0.0, v_line=-0.5)


        if self.ax_trajectory is not None:
            Rt_arr = logger.hist.data['Rt']
            St_pos_arr = logger.hist.data['St_pos']
            St_vel_arr = logger.hist.data['St_vel']
            At_arr = logger.hist.data['At']
            done = logger.hist.data['done']

            i = current_total_step
            Rt = Rt_arr[ max(0, i-self.disp_len) : i + 1 ]
            St_pos = St_pos_arr[ max(0, i-self.disp_len) : i + 1 ]
            St_vel = St_vel_arr[ max(0, i-self.disp_len) : i + 1 ]
            At = At_arr[ max(0, i-self.disp_len) : i + 1 ]

            self.ax_trajectory.clear()
            plot_trajectory_2d(self.ax_trajectory, 
                St_pos, St_vel, At, extent, h_line=0.0, v_line=-0.5)

        # if ax_stats is not None:
        #     ax_stats.clear()
        #     i = current_total_step

        #     t_steps = logger.agent.total_steps[0:i:1]
        #     ser_e_rand = logger.agent.data['e_rand'][0:i:1]
        #     ser_rand_act = logger.agent.data['rand_act'][0:i:1]
        #     ser_mem_size = logger.agent.data['mem_size'][0:i:1]

        #     arr = logger.agent.data['rand_act'][max(0, i-1000):i]
        #     nz = np.count_nonzero(arr)
        #     print('RAND: ', nz, ' / ', len(arr))

        #     # ax_stats.plot(t_steps, ser_e_rand, label='e_rand', color='red')
        #     ax_stats.plot(t_steps, ser_rand_act, label='rand_act', color='blue')
        #     ax_stats.legend()

        if self.ax_memory is not None:
            self.ax_memory.clear()
            # plot_trajectory_2d(self.ax_memory,
            #     self.hist_St[-1000:,0],
            #     self.hist_St[-1000:,1],
            #     self.hist_At[-1000:,0],
            #     extent, h_line=0.0, v_line=-0.5)

            self.ax_memory.plot(self.hist_done * 100, color='green')
            self.ax_memory.plot(self.hist_error, color='blue')
            
            self.ax_memory.set_ylim([0, 100])


        if self.ax_q_series is not None:
            self.ax_q_series.clear()
            plot_q_series(self.ax_q_series,
                self.ser_X[-50:],
                self.ser_E0[-50:],
                self.ser_E1[-50:],
                self.ser_E2[-50:])

        if self.ax_reward is not None:
            self.ax_reward.clear()

            epsumm_end = logger.epsumm.data['end']    # list of ints
            epsumm_rew = logger.epsumm.data['reward'] # list of floats

            ep_ends = []
            ep_rewards = []
            ep_avg_rew = []
            for i in range(len(epsumm_end)):
                ep_ends.append(epsumm_end[i])
                ep_rewards.append(epsumm_rew[i])
                ith_chunk = epsumm_rew[max(0, i-49):i+1]
                ep_avg_rew.append(sum(ith_chunk) / len(ith_chunk))

            self.ax_reward.plot(ep_ends, ep_rewards, color='black', marker='o', markerfacecolor='None')
            self.ax_reward.plot(ep_ends, ep_avg_rew, color='gray', marker='o', markerfacecolor='None')

def plot_q_val_wireframe(ax, q_val, extent, labels):
    """Plot 2d q_val array on 3d wireframe plot.
    
    Params:
        ax - axis to plot on
        q_val - 2d numpy array as follows:
                1-st dim is X, increasing as indices grow
                2-nd dim is Y, increasing as indices grow
        extent - [x_min, x_max, y_min, y_max]
        labels - [x_label, y_label, z_label]
    """

    assert len(extent) == 4
    assert len(labels) == 3

    x_min, x_max, y_min, y_max = extent
    x_label, y_label, z_label = labels

    x_size = q_val.shape[0]
    y_size = q_val.shape[1]
    x_space = np.linspace(x_min, x_max, x_size)
    y_space = np.linspace(y_min, y_max, y_size)

    Y, X = np.meshgrid(y_space, x_space)
    
    ax.plot_wireframe(X, Y, q_val)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)


def plot_q_val_imshow(ax, q_val, extent, h_line, v_line):
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    ax.imshow(q_val.T, extent=extent, 
        aspect='auto', origin='lower',
        interpolation='gaussian')

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')


def plot_policy(ax, q_val, extent, h_line, v_line):
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    x_size = q_val.shape[0]
    y_size = q_val.shape[1]
    x_space = np.linspace(x_min, x_max, x_size)
    y_space = np.linspace(y_min, y_max, y_size)

    data_draw_x = []
    data_draw_y = []
    data_a0_x = []
    data_a0_y = []
    data_a1_x = []
    data_a1_y = []
    data_a2_x = []
    data_a2_y = []

    max_act = np.argmax(q_val, axis=2)

    for xi in range(x_size):
        for yi in range(y_size):

            x = x_space[xi]
            y = y_space[yi]

            if q_val[xi, yi, 0] == q_val[xi, yi, 1] == -100.0:
                data_draw_x.append(x)
                data_draw_y.append(y)

            elif max_act[xi, yi] == 0:
                data_a0_x.append(x)
                data_a0_y.append(y)
            elif max_act[xi, yi] == 1:
                data_a1_x.append(x)
                data_a1_y.append(y)
            elif max_act[xi, yi] == 2:
                data_a2_x.append(x)
                data_a2_y.append(y)
            else:
                raise ValueError('bad')

    ax.scatter(data_draw_x, data_draw_y, color='gray', marker='.')
    ax.scatter(data_a0_x, data_a0_y, color='red', marker='.')
    ax.scatter(data_a1_x, data_a1_y, color='blue', marker='.')
    ax.scatter(data_a2_x, data_a2_y, color='green', marker='.')

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def plot_trajectory_2d(ax, x_arr, y_arr, act_arr, extent, h_line, v_line):
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    data_a0_x = []
    data_a0_y = []
    data_a1_x = []
    data_a1_y = []
    data_a2_x = []
    data_a2_y = []

    for i in range(len(x_arr)):
        if act_arr[i] == 0:
            data_a0_x.append(x_arr[i])
            data_a0_y.append(y_arr[i])
        elif act_arr[i] == 1:
            data_a1_x.append(x_arr[i])
            data_a1_y.append(y_arr[i])
        elif act_arr[i] == 2:
            data_a2_x.append(x_arr[i])
            data_a2_y.append(y_arr[i])
        elif act_arr[i] is None:
            # terminal state
            pass
        else:
            print('act_arr[i] = ', act_arr[i])
            raise ValueError('bad')

    ax.scatter(data_a0_x, data_a0_y, color='red', marker=',', lw=0, s=1)
    ax.scatter(data_a1_x, data_a1_y, color='blue', marker=',', lw=0, s=1)
    ax.scatter(data_a2_x, data_a2_y, color='green', marker=',', lw=0, s=1)

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

def plot_q_series(ax, t_steps, ser_0, ser_1, ser_2):

    # x = list(range(len(approx._q_back)))

    ax.plot(t_steps, ser_0, color='red')
    ax.plot(t_steps, ser_1, color='blue')
    ax.plot(t_steps, ser_2, color='green')


if __name__ == '__main__':
    main()