



import unittest

import gym
import pandas as pd

import gym_trading
from gym_trading.utils import Visualize
from gym_trading.utils.decorator import print_time


class TrendFollowingTestCases(unittest.TestCase):

    @print_time
    def test_time_event_env(self):
        viz = Visualize(
            columns=['midpoint', 'buys', 'sells', 'inventory', 'realized_pnl'],
            store_historical_observations=True)

        df = pd.read_csv("../../training/to_visualize")
        viz.plot_episode_history_plotly(history=df, save_filename="../../training/plots/test-hft-plot-trade-history")
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
