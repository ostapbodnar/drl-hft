from gym.envs.registration import register

from gym_trading.envs.high_frequency_trading import HighFrequencyTrading

register(
    id=HighFrequencyTrading.id,
    entry_point='gym_trading.envs:HighFrequencyTrading',
    max_episode_steps=1000000,
    nondeterministic=False
)
