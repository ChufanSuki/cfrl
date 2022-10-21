import logging

def train_agent(agent, env, steps, outdir, max_episode_len=None, logger=None, step_offset=0):
    """train agent 

    Args:
        agent (Agent): agent to train
        env (Env): environment
        steps (int): training steps
        outdir (str): path to output directory
        max_episode_len (int, optional): max steps for each episode. Defaults to None.
        logger (Logger, optional): logger. Defaults to None.
        step_offset (int, optional): step offset to start. Defaults to 0.
    """
    logger = logger or logging.getLogger(__name__)

    epsiode_idx = 0
    epsiode_r = 0
    epsiode_len = 0
    epsiode_end = None

    obs = env.reset()

    t = step_offset

    try:
        while t < steps:
            action = agent.act(obs)
            obs, r, done, info = env.step(action)
            t += 1
            epsiode_r += r
            epsiode_len += 1
            reset = epsiode_len == max_episode_len
            agent.observe(obs, r, done, reset)
            epsiode_end = done or reset or t == steps

            if epsiode_end:
                logger.info("episode: %d R: %d", epsiode_idx, epsiode_r)
                epsiode_idx += 1
                if t < steps:
                    epsiode_r = 0
                    epsiode_len = 0
                    obs = env.reset()
    except (Exception, KeyboardInterrupt):
        raise

