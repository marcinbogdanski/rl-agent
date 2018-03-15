# RL Agent

Mini Reinforcement Learning Library

## Short Info

This is a mini reinforcement learning repo implementing simple agent for OpenAI gym.

This is work in progress.

Currently available agents:

**AgentQ** - e-greedy agent with q-function approximation
 * Algorithm - not-configurable atm., currenty **SARSA**, other to be added later (e.g. Monte-Carlo, Eligibility Traces, Q-Learning)
 * QFunction - pick from: **QFunctTabular**, **QFunctAggregate**, **QFunctTiles**
 * Policy - must be **PolicyEpsGreedy**
 
**AgentDQN** - DQN agent, does everything as AgentQ, but includes reply memory as well
 * Algorithm - not-configurable, always **Q-Learning**
 * QFunction - should be non-linear **QFunctKeras**, specify keras model separately, other to be added later (?)
 * Policy - must be **PolicyEpsGreedy**
 
**AgentActorCritic** - similar AgentQ, allows for different policies and state-value (V-Function) approximators
 * Batching - not implemented - allow for collecting multiple episodes before learning
 * Algorithm - non-configurable, currently **Monte-Carlo**, possibly add other later or even allow different alg. for actor and critic
 * VFunction - not implemented, later add: **VFunctTabular**, **VFunctAggregate**, **VFunctTiles**
 * QFunction - pick from: **QFunctTabular**, **QFunctAggregate**, **QFunctTiles**
 * Policy - pick from: **PolicyTabularCat**, **PolicyTabularCont**, add Tiles and non-linear policies later

Future agents (possibly?)

**AgentA2C** - same as AgentActorCritic, but instead of batching it runs multiple agents in parallel on different copies of environment

## Alternatives

If you are looking for more mature RL codebase, here are few choices:
 * [OpenAI baselines](https://github.com/openai/baselines)
 * [Deepmind Lab](https://github.com/deepmind/lab)
 * [RLLab](https://github.com/rll/rllab)
 * [TensorForce](https://github.com/reinforceio/tensorforce/tree/master/tensorforce)
 * [keras-rl](https://github.com/matthiasplappert/keras-rl/tree/master/rl/agents)
 
