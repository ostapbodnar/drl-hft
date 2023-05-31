import argparse
import sys

import gym
import torch

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
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
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
    print(f"Testing {actor_model}", flush=True)

    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    act_dim = env.action_space.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy = CnnLstmTwoHeadNN(act_dim, device)

    policy.load_state_dict(torch.load(actor_model))


def main(args):
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
        fitting_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_min_labeled.csv',
        testing_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_min_labeled.csv',
        max_position=20,
        window_size=100,
        seed=2,
        action_repeats=10,
        training=True,
        format_3d=False,
        reward_type='trade_completion',
        ema_alpha=None,
        shuffle_on_reset=False,
        **ppo_agent_parameters
    )
    print(f"**********\n{env_config}\n**********")

    env: gym_trading.HighFrequencyTrading = gym.make(**env_config)

    if args.mode == 'train':
        train_model(env=env, hyperparameters=ppo_agent_parameters, actor_model="ppo_actor.pth",
                    critic_model="ppo_critic.pth")
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
