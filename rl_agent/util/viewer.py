import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import rl_agent as rl





def main():
    """This is separate program used to parse and plot .log created by Logger.

    This was NOT TESTED in a while and probably doesn't work at the moment.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='Data log filename')
    args = parser.parse_args()

    logger = rl.logger.Logger()
    logger.load(args.filename)

    print()
    print(logger)
    print(logger.env)
    print(logger.agent)
    print(logger.q_val)
    print(logger.hist)
    print(logger.approx)

    fig = plt.figure()
    ax_qmax_wf = fig.add_subplot(2,4,1, projection='3d')
    ax_qmax_im = fig.add_subplot(2,4,2)
    ax_policy = fig.add_subplot(2,4,3)
    ax_trajectory = fig.add_subplot(2,4,4)
    ax_stats = None # fig.add_subplot(165)
    ax_memory = None # fig.add_subplot(2,1,2)
    ax_q_series = None # fig.add_subplot(155)
    ax_reward = fig.add_subplot(2,1,2)
    plotter = rl.logger.Plotter(  realtime_plotting=True,
                                  plot_every=1000,
                                  disp_len=1000,
                                  ax_qmax_wf=ax_qmax_wf,
                                  ax_qmax_im=ax_qmax_im,
                                  ax_policy=ax_policy,
                                  ax_trajectory=ax_trajectory,
                                  ax_stats=ax_stats,
                                  ax_memory=ax_memory,
                                  ax_q_series=ax_q_series,
                                  ax_reward=ax_reward  )

    for total_step in range(0, len(logger.hist.total_steps)):
        print(total_step)

        plotter.process(logger, total_step)
        res = plotter.conditional_plot(logger, total_step)
        if res:
            plt.pause(0.1)

    plt.show()


if __name__ == '__main__':
    main()