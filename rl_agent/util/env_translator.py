

class EnvTranslator:
    def __init__(self,
                env,
                observation_space=None,
                observation_translator=None,
                action_space=None,
                action_translator=None,
                reward_translator=None):
        """Translate observations, actions and rewards between Env and Agent.

        If observation_space or observation_translator is specified, then
        both have to be specified. Alternatively both can be set to None if
        no translation is required. Same for actions.

        Params:
            env - environment to be wrapped
            observation_space: obs space to be usesd by the Agent, can be None
            observation_translator: function that translates env -> agent
            action_space: action space to be used by the Agent
            action_translator: function that translates actions agent -> env
            reward_translator: reward engineering):
        """

        if observation_space is not None and observation_translator is None:
            raise ValueError('Must specify both observation_space and observation_translator')
        if observation_space is None and observation_translator is not None:
            raise ValueError('Must specify both observation_space and observation_translator')
       
        if action_space is not None and action_translator is None:
            raise ValueError('Must specify both action_space and action_translator')
        if action_space is None and action_translator is not None:
            raise ValueError('Must specify both action_space and action_translator')
       
        self.env = env
        self.observation_space = observation_space if observation_space is not None else env.observation_space
        self.observation_translator = observation_translator
        self.action_space = action_space if action_space is not None else env.action_space
        self.action_translator = action_translator
        self.reward_translator = reward_translator

    def reset(self):
        """Wrap env.reset()"""
        observation = self.env.reset()
        if self.observation_translator is not None:
            observation = self.observation_translator(observation)
            assert self.observation_space.contains(observation)
        return observation

    def step(self, action):
        """Wrap env.step()"""
        if self.action_translator is not None:
            assert self.action_space.contains(action)
            action = self.action_translator(action)
            assert self.env.action_space.contains(action)
        
        observation, reward, done, info = self.env.step(action)

        if self.observation_translator is not None:
            observation = self.observation_translator(observation)
            assert self.observation_space.contains(observation)

        if self.reward_translator is not None:
            reward = self.reward_translator(observation, action, reward)
        
        return observation, reward, done, info

    def seed(self, seed):
        self.env.seed(seed)

    def render(self):
        self.env.render()