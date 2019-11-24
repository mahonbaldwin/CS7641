from gym.envs.registration import register

register(
      id='casino-v0',
      entry_point='gym_casino.envs:CasinoEnv',
)