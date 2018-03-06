import numpy as np

class HistoryData:
    """One piece of agent trajectory (obs, rew, act, done)"""
    def __init__(self, total_step, observation, reward, done):
        assert isinstance(total_step, int)
        assert total_step >= 0
        assert isinstance(observation, np.ndarray)
        self.total_step = total_step
        self.observation = observation
        self.reward = reward
        self.action = None
        self.done = done

    def __str__(self):
        return 'obs={0}, rew={1} done={2}   act={3}'.format(
            self.observation, self.reward, self.done, self.action)


class EpisodeData:
    """Episode summary"""
    def __init__(self, start, end, length, total_reward):
        assert end-start == length-1
        self.start = start
        self.end = end
        self.length = length
        self.total_reward = total_reward

class AgentBase:
    """Under construction"""


    def __init__(self,
        state_space,
        action_space,
        discount,
        start_learning_at):

        self._state_space = state_space
        self._action_space = action_space
        self._discount = discount  # usually gamma in literature
        self._start_learning_at = start_learning_at




