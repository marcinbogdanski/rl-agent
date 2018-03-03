import numpy as np
import matplotlib.pyplot as plt

import pdb



class Plotter():
    def __init__(self, realtime_plotting, plot_every, disp_len, 
        figures, ax_qmax_wf, ax_qmax_im, ax_policy,
        ax_trajectory, ax_stats, ax_memory, ax_q_series, ax_reward):

        self.realtime_plotting = realtime_plotting
        self.plot_every = plot_every
        self.disp_len = disp_len

        self.figures = figures

        self.ax_qmax_wf = ax_qmax_wf
        self.ax_qmax_im = ax_qmax_im
        self.ax_policy = ax_policy
        self.ax_trajectory = ax_trajectory
        self.ax_stats = ax_stats
        self.ax_memory = ax_memory
        self.ax_q_series = ax_q_series
        self.ax_reward = ax_reward

        
    def conditional_plot(self, logger, current_total_step):
        if current_total_step % self.plot_every == 0 and self.realtime_plotting:
            self.plot(logger, current_total_step)
            return True
        else:
            return False

    def plot(self, logger, current_total_step):

        extent = (-1.2, 0.5, -0.07, 0.07)

        # print('---')
        # print(current_total_step % step_span == 0)
        # print(q_val is not None)

        q_val, _, _, q_val_step = logger.q_val.get_last('q_val')
        if q_val is not None:
            q_max = np.max(q_val, axis=2)

            if self.ax_qmax_wf is not None:
                self.ax_qmax_wf.clear()
                self.ax_qmax_wf.set_title('q_max: ' + str(q_val_step))
                plot_q_val_wireframe(self.ax_qmax_wf, q_max,
                    extent, ('pos', 'vel', 'q_max'), color='gray', alpha=1.0)

            if self.ax_qmax_im is not None:
                self.ax_qmax_im.clear()
                plot_q_val_imshow(self.ax_qmax_im, q_max,
                    extent, h_line=0.0, v_line=-0.5)
                self.ax_qmax_im.set_yticklabels([])
                self.ax_qmax_im.set_xticklabels([])
            
            if self.ax_policy is not None:
                self.ax_policy.clear()
                plot_policy(self.ax_policy, q_val,
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


        if self.ax_memory is not None:
            hist_St, _, _, hist_step = logger.memory.get_last('hist_St')
            hist_At, _, _, _ = logger.memory.get_last('hist_At')
            hist_Rt_1, _, _, _ = logger.memory.get_last('hist_Rt_1')
            hist_St_1, _, _, _ = logger.memory.get_last('hist_St_1')
            hist_done, _, _, _ = logger.memory.get_last('hist_done')
            hist_error, _, _, _ = logger.memory.get_last('hist_error')

            self.ax_memory.clear()
            self.ax_memory.set_title('mem: ' + str(hist_step))
            
            self.ax_memory.plot(hist_done * 100, color='green')
            self.ax_memory.plot(hist_error, color='blue')
            
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

            self.ax_reward.plot(
                ep_ends, ep_rewards, color='black', 
                marker='o', markerfacecolor='None')
            self.ax_reward.plot(
                ep_ends, ep_avg_rew, color='gray',
                marker='o', markerfacecolor='None')

        for fig in self.figures:
            fig.canvas.draw()
        plt.pause(0.001)

def plot_q_val_wireframe(ax, q_val, extent, labels, color, alpha):
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
    
    ax.plot_wireframe(X, Y, q_val, color=color, alpha=alpha)

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
