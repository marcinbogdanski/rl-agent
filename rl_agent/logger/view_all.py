import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os

import rl_agent as rl


import pdb


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('dirname', help='Data log filename')
    args = parser.parse_args()

    filenames = [f for f in os.listdir(args.dirname)]

    

    fig = plt.figure()
    ax_reward = fig.add_subplot(1, 1, 1)


    for filename in filenames:
        filepath = os.path.join(args.dirname, filename)
        if os.path.isfile(filepath) and filepath.endswith('.log'):
            print(filepath)

            logger = rl.logger.Logger()
            logger.load(filepath)

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

            ax_reward.plot(ep_ends, ep_rewards, marker='.', markerfacecolor='None', label=filename)
            #ax_reward.plot(ep_ends, ep_avg_rew, color='gray', marker='.', markerfacecolor='None')

            plt.legend()
            
            plt.pause(0.1)

    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()