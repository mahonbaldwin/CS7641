from gym.envs.registration import register

register(
    id='Casino-v0',
    entry_point='gym_casino.envs:CasinoEnv',
)