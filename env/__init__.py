from gym.envs.registration import register

register(
    id='CartPoleContinuous-v0',
    entry_point='env.cartpole_continuous:CartPoleContinuousEnv',
    timestep_limit=200,
    reward_threshold=195.0,
)
