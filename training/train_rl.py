import argparse
import datetime
import sys
import time

import gym
import numpy as np
import torch
from torch.distributions import Categorical

import gym_trading
from agent.nn_model import CnnLstmTwoHeadNN
from agent.ppo import PPO


def train_model(env, hyperparameters, actor_model, critic_model):
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=CnnLstmTwoHeadNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model, map_location=model.device))
        model.critic.load_state_dict(torch.load(critic_model, map_location=model.device))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)


def test_model(env, actor_model):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    print(f"Testing {actor_model}", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = CnnLstmTwoHeadNN(env.action_space.n, device)

    policy.load_state_dict(torch.load(actor_model, map_location=device))

    reward_history = []
    n_steps = 0

    start_time = time.time()
    observation = env.reset()
    done = False
    score = 0

    actions_tracker = dict()
    while not done:
        action_probs = policy(torch.tensor(np.asarray([observation]), device=device))
        dist = Categorical(action_probs)
        action = dist.sample().item()

        actions_tracker[action] = actions_tracker.setdefault(action, 0) + 1

        observation_, reward, done, _, info = env.step(action)
        n_steps += 1
        score += reward
        observation = observation_
        reward_history.append(reward)
    avg_reward = np.mean(reward_history[-100:])

    print('score %.1f' % score, 'avg score %.1f' % avg_reward, 'time_steps', n_steps,
          "time", datetime.timedelta(seconds=time.time() - start_time))
    print(actions_tracker, env.action_to_levels_mapping)

    print(f"Max reward: {max(reward_history)}\nMin reward: {min(reward_history)}")
    print(f"Agent completed {env.broker.total_trade_count} trades")

    env.plot_trade_history('plots/hft-plot-trade-history-eval')
    env.reset()


def main(args):
    training = args.mode == 'train'
    ppo_agent_parameters = dict(
        timesteps_per_batch=2000,
        max_timesteps_per_episode=500,
        gamma=0.99,
        n_updates_per_iteration=20,
        lr=3e-4,
        clip=0.2,
        render=False,
        save_freq=100
    )
    env_config = dict(
        id=gym_trading.envs.HighFrequencyTrading.id,
        symbol='BTC_USDT',
        fitting_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_min_labeled.csv',
        testing_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_min_labeled.csv',
        max_position=20,
        window_size=100,
        seed=2,
        action_repeats=10,
        add_target_loss_to_reward=True,
        training=training,
        format_3d=False,
        reward_type='trade_completion',
        ema_alpha=None,
        shuffle_on_reset=False,
        **ppo_agent_parameters
    )
    print(f"**********\n{env_config}\n**********")

    env: gym_trading.HighFrequencyTrading = gym.make(**env_config)

    if training:
        train_model(env=env, hyperparameters=ppo_agent_parameters, actor_model=args.actor_model,
                    critic_model=args.critic_model)
    else:
        test_model(env=env, actor_model=args.actor_model)


def get_args():
    """
        Description:
        Parses arguments at command line.

        Parameters:
            None

        Return:
            args - the arguments parsed
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    main(args)
