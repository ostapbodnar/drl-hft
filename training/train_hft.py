import datetime
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

import gym_trading
from agent.custom_agent import Agent
from gym_trading.envs.base_environment import Observation


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    fig = px.line(x, running_avg, title='Running average of previous 100 scores')
    fig.write_image(figure_file)


if __name__ == '__main__':
    config = dict(
        id=gym_trading.envs.HighFrequencyTrading.id,
        symbol='BTC_USDT',
        fitting_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_1618028600000_1618220350000.csv',
        testing_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_1618028600000_1618220350000.csv',
        # testing_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_1618028600000_1618220350000.csv',
        max_position=20,
        window_size=1,
        seed=1,
        action_repeats=5,
        training=True,
        format_3d=True,
        reward_type='default',
        ema_alpha=None,
    )
    print(f"**********\n{config}\n**********")

    env: gym_trading.HighFrequencyTrading = gym.make(**config)

    N = 20
    batch_size = 32
    n_epochs = 1
    agent = Agent(mlp_hidden_size=64,
                  num_classes=env.action_space.n,
                  batch_size=batch_size, n_epochs=n_epochs)
    n_games = 1

    figure_file = 'plots/hft-score.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        start_time = time.time()
        observation = env.reset()
        done = False
        score = 0

        actions_tracker = dict()
        while not done:
            kline, lob = [o[:5] for o in observation], [o[5:] for o in observation]
            action, prob, val = agent.choose_action(kline, lob)

            if action in actions_tracker:
                actions_tracker[action] += 1
            else:
                actions_tracker[action] = 1

            observation_, reward, done, _, info = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(kline[0], lob[0], action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters, "time",
              datetime.timedelta(seconds=time.time() - start_time))
        print(actions_tracker, env.action_to_levels_mapping)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)

    print(f"Max reward: {max(score_history)}\nMin reward: {min(score_history)}")
    print(f"Agent completed {env.broker.total_trade_count} trades")
    # Visualize results
    # env.plot_observation_history('plots/hft-plot-observation-history.png')
    env.plot_trade_history('plots/hft-plot-trade-history.png')
