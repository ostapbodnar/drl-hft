import datetime
import time

import gym
import numpy as np

import gym_trading
from agent.custom_agent import Agent
from gym_trading.envs.base_environment import Observation

if __name__ == '__main__':
    print("EVALUATION")
    config = dict(
        id=gym_trading.envs.HighFrequencyTrading.id,
        symbol='BTC_USDT',
        fitting_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_1618028600000_1618220350000.csv',
        testing_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_1618028600000_1618220350000.csv',
        # testing_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_1618028600000_1618220350000.csv',
        max_position=20,
        window_size=100,
        seed=1,
        action_repeats=10,
        training=False,
        format_3d=False,
        reward_type='trade_completion',
        ema_alpha=None,
    )
    print(f"**********\n{config}\n**********")

    env: gym_trading.HighFrequencyTrading = gym.make(**config)

    batch_size = 128
    n_epochs = 1
    agent = Agent(mlp_hidden_size=64,
                  num_classes=env.action_space.n,
                  batch_size=batch_size, n_epochs=n_epochs)

    agent.load_models()

    best_score = env.reward_range[0]
    reward_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    start_time = time.time()
    observation = env.reset()
    done = False
    score = 0

    actions_tracker = dict()
    while not done:
        kline, lob = np.array([observation[:, :5]]), np.array([observation[:, 5:]])
        action, prob, val = agent.choose_action(kline, lob)

        if action in actions_tracker:
            actions_tracker[action] += 1
        else:
            actions_tracker[action] = 1

        observation_, reward, done, _, info = env.step(action)
        n_steps += 1
        score += reward
        observation = observation_
        reward_history.append(reward)
    avg_reward = np.mean(reward_history[-100:])

    print('score %.1f' % score, 'avg score %.1f' % avg_reward, 'time_steps', n_steps,
          "time", datetime.timedelta(seconds=time.time() - start_time))
    print(actions_tracker, env.action_to_levels_mapping)
    x = [i + 1 for i in range(len(reward_history))]

    print(f"Max reward: {max(reward_history)}\nMin reward: {min(reward_history)}")
    print(f"Agent completed {env.broker.total_trade_count} trades")

    env.plot_trade_history('plots/hft-plot-trade-history-eval.png')
    env.reset()
