import numpy as np
import matplotlib.pyplot as plt

import pdb

# TODO: think if this is correct architecture, how to improve?
# TODO: improve documentaiton
# TODO: add more assert statements all around

class Plotter():
    def __init__(self, realtime_plotting, plot_every, disp_len, nb_actions,
        figures, ax_qmax_wf, ax_qmax_im, ax_policy,
        ax_trajectory, ax_stats, ax_memory, ax_q_series, ax_reward):
        """Used for plotting stuff. Uses Logger logs to get data.

        This is tightly integrated with how Logger works, this should be changed
        in the future to allow more flexibility.
    
        This class is probably going to be changed a lot in the future

        IMPORTANT:
        All Q-plots assume 2D continous state space with categorical actions.

        Params:
            realtime_plotting: plot as data comes in, TODO: assume always yes?
            plot_every (int): plot every t time steps, skip otherwise
            disp_len: how many most recent time steps to plot
            nb_actions: required for policy plot, TODO: remove
            figures: pass list of all figures to refresh after plotting is done
            ax_qmax_wf: 3d wireframe plot for Q-max, assumed projection='3d'
            ax_qmax_im: 2d plot for Q-max
            ax_policy: plot policy, colors corresponding to different actions
            ax_trajectory: agent trajectory over last disp_len steps
            ax_stats: not used at the moment?
            ax_memory: full memory dump for DQN agents
            ax_q_series: not used
            ax_reward: cumulative reward for each episode
        """

        self.realtime_plotting = realtime_plotting
        self.plot_every = plot_every
        self.disp_len = disp_len
        self.nb_actions = nb_actions

        self.extent = None
        self.h_line = None
        self.v_line = None

        self.figures = figures

        self.ax_qmax_wf = ax_qmax_wf
        self.ax_qmax_im = ax_qmax_im
        self.ax_policy = ax_policy
        self.ax_trajectory = ax_trajectory
        self.ax_stats = ax_stats
        self.ax_memory = ax_memory
        self.ax_q_series = ax_q_series
        self.ax_reward = ax_reward

    def set_state_action_spaces(self, low, high, h_line, v_line):
        """Set state-space range

        Params:
            low (array): low values for each state-space dimension
            high (array): high values for state-space
            h_line: draw extra horizontal line on plots
            v_line: draw extra vertical line on plots
        """
        self.extent = (low[0], high[0], low[1], high[1])
        self.h_line = h_line
        self.v_line = v_line

        
    def conditional_plot(self, logger, current_total_step):
        """Check if should plot in this time step and possibly plot"""
        if current_total_step % self.plot_every == 0 and self.realtime_plotting:
            self.plot(logger, current_total_step)
            return True
        else:
            return False

    def plot(self, logger, current_total_step):
        """Plot everything, and update figures

        plt.pause() at the end of this funcion can CRASH program if
        pdb is used somewhere else in the code.
        """

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
                    self.extent, ('pos', 'vel', 'q_max'), color='gray', alpha=1.0)

            if self.ax_qmax_im is not None:
                self.ax_qmax_im.clear()
                plot_q_val_imshow(self.ax_qmax_im, q_max,
                    self.extent, h_line=self.h_line, v_line=self.v_line)
                self.ax_qmax_im.set_yticklabels([])
                self.ax_qmax_im.set_xticklabels([])
            
            if self.ax_policy is not None:
                self.ax_policy.clear()
                plot_policy(self.ax_policy, q_val,
                    self.extent, h_line=self.h_line, v_line=self.v_line)


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
                St_pos, St_vel, At, self.nb_actions, 
                self.extent, h_line=self.h_line, v_line=self.v_line)


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

            #
            #   Average episodic reward
            #
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
        ax: axis to plot on
        q_val: 2d numpy array as follows:
               1-st dim is X, increasing as indices grow
               2-nd dim is Y, increasing as indices grow
        extent: [x_min, x_max, y_min, y_max]
        labels: [x_label, y_label, z_label]
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
    """Plot Q-val on 2d heat-map like plot"""
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    ax.imshow(q_val.T, extent=extent, 
        aspect='auto', origin='lower',
        interpolation='gaussian')

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')


def plot_policy(ax, q_val, extent, h_line, v_line):
    """Plot colorfull dots on 2d space corresponding to actions"""
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    x_size = q_val.shape[0]
    y_size = q_val.shape[1]
    x_space = np.linspace(x_min, x_max, x_size)
    y_space = np.linspace(y_min, y_max, y_size)

    nb_actions = q_val.shape[-1]

    data_draw_x = []
    data_draw_y = []
    data_act_x = []
    data_act_y = []
    for i in range(nb_actions):
        data_act_x.append([])
        data_act_y.append([])

    max_act = np.argmax(q_val, axis=2)

    for xi in range(x_size):
        for yi in range(y_size):

            x = x_space[xi]
            y = y_space[yi]

            all_equal = False
            value = q_val[xi, yi, 0]
            for i in range(1, nb_actions):
                if q_val[xi, yi, i] != value:
                    all_equal = False
                    break

            if all_equal:
                data_draw_x.append(x)
                data_draw_y.append(y)
            else:
                action = max_act[xi, yi]
                data_act_x[action].append(x)
                data_act_y[action].append(y)

    if nb_actions == 2:
        colors = ('red', 'green')
    elif nb_actions == 3:
        colors = ('red', 'blue', 'green')
    elif nb_actions == 5:
        colors = ('red', 'orange', 'blue', 'cyan', 'darkgreen')

    ax.scatter(data_draw_x, data_draw_y, color='gray', marker='.')
    for i in range(nb_actions):
        ax.scatter(data_act_x[i], data_act_y[i], color=colors[i], marker='.', lw=0, s=30)

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])


def plot_trajectory_2d(ax, x_arr, y_arr, act_arr, nb_actions, extent, h_line, v_line):
    """Plot recent agent trajectory

    For mountain car this will plot circles mostly
    For pendulum it will also plot circle-curvle like paths
    """
    assert len(extent) == 4

    x_min, x_max, y_min, y_max = extent

    data_act_x = []
    data_act_y = []
    for i in range(nb_actions):
        data_act_x.append([])
        data_act_y.append([])

    for i in range(len(x_arr)):
        action = act_arr[i]

        if action is None:
            # terminal state
            pass
        else:
            data_act_x[action].append(x_arr[i])
            data_act_y[action].append(y_arr[i])

    if nb_actions == 2:
        colors = ('red', 'green')
    elif nb_actions == 3:
        colors = ('red', 'blue', 'green')
    elif nb_actions == 5:
        colors = ('darkred', 'red', 'blue', 'green', 'darkgreen')

    for i in range(nb_actions):
        ax.scatter(data_act_x[i], data_act_y[i], color=colors[i], marker=',', lw=0, s=1)

    ax.plot([x_min, x_max], [h_line, h_line], color='black')
    ax.plot([v_line, v_line], [y_min, y_max], color='black')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

def plot_q_series(ax, t_steps, ser_0, ser_1, ser_2):

    # x = list(range(len(approx._q_back)))

    ax.plot(t_steps, ser_0, color='red')
    ax.plot(t_steps, ser_1, color='blue')
    ax.plot(t_steps, ser_2, color='green')
