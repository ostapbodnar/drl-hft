import unittest

import gym

import gym_trading
from gym_trading.utils.decorator import print_time


class MarketMakerTestCases(unittest.TestCase):

    @print_time
    def test_time_event_env(self):
        config = dict(
            id=gym_trading.envs.HighFrequencyTrading.id,
            symbol='BTC_USDT',
            fitting_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021.csv',
            testing_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021.csv',
            max_position=10,
            window_size=5,
            seed=1,
            action_repeats=5,
            training=False,
            format_3d=True,
            reward_type='default',
            ema_alpha=None,
        )
        print(f"**********\n{config}\n**********")

        env = gym.make(**config)
        done = gym_trading.envs.test_env_loop(env=env)
        _ = env.reset()
        self.assertEqual(True, done)


if __name__ == '__main__':
    unittest.main()
