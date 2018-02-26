

def train_agent(env, agent, total_steps):

    agent.reset()

    done = True

    while True:

        #   ---------------------------------
        #   ---   time step starts here   ---

        # environment step, reset if required
        if done:
            obs, reward, done = env.reset(), None, False
        else:
            obs, reward, done, _ = env.step(action)

        # save obs to agent trajectory
        agent.observe(obs, reward, done)

        # learn from trajectory
        agent.learn()
        
        # select action accorrding to policy and save to agent trajectory
        # use pick_action() if you don't want to save action to trajecotry
        action = agent.take_action(obs)

        # log, callbacks, housekeeping
        agent.next_step(done)

        if agent.total_step > total_steps:
            break     

        #   ---    time step ends here    ---
        #   ---------------------------------