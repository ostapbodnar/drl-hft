import argparse
import sys

import gym
import torch

import gym_trading
from agent.agent_for_beginers import PPO
from agent.nn_model import CnnLstmTwoHeadNN


def train(env, hyperparameters, actor_model, critic_model):
    print(f"Training", flush=True)

    # Create a model for PPO.
    model = PPO(policy_class=CnnLstmTwoHeadNN, env=env, **hyperparameters)

    # Tries to load in an existing actor/critic model to continue training on
    if actor_model != '' and critic_model != '':
        print(f"Loading in {actor_model} and {critic_model}...", flush=True)
        model.actor.load_state_dict(torch.load(actor_model))
        model.critic.load_state_dict(torch.load(critic_model))
        print(f"Successfully loaded.", flush=True)
    elif actor_model != '' or critic_model != '':  # Don't train from scratch if user accidentally forgets actor/critic model
        print(
            f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    # Train the PPO model with a specified total timesteps
    # NOTE: You can change the total timesteps here, I put a big number just because
    # you can kill the process whenever you feel like PPO is converging
    model.learn(total_timesteps=200_000_000)


def _test(env, actor_model):
    """
        Tests the model.

        Parameters:
            env - the environment to test the policy on
            actor_model - the actor model to load in

        Return:
            None
    """
    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # Extract out dimensions of observation and action spaces
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    policy = CnnLstmTwoHeadNN(act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    # eval_policy(policy=policy, env=env, render=True)


def main(args):
    hyperparameters = {
        'timesteps_per_batch': 2000,
        'max_timesteps_per_episode': 500,
        'gamma': 0.99,
        'n_updates_per_iteration': 20,
        'lr': 3e-4,
        'clip': 0.2,
        'render': False,
        'save_freq': 100
    }

    config = dict(
        id=gym_trading.envs.HighFrequencyTrading.id,
        symbol='BTC_USDT',
        fitting_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_min_labeled.csv',
        testing_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_min_labeled.csv',
        # testing_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_min_1618028600000_1618220350000.csv',
        max_position=20,
        window_size=100,
        seed=2,
        action_repeats=10,
        training=True,
        format_3d=False,
        reward_type='trade_completion',
        ema_alpha=None,
        shuffle_on_reset=False,
        **hyperparameters
    )
    print(f"**********\n{config}\n**********")

    env: gym_trading.HighFrequencyTrading = gym.make(**config)

    # Train or test, depending on the mode specified
    if args.mode == 'train':
        train(env=env, hyperparameters=hyperparameters, actor_model="ppo_actor-labeled.pth", critic_model="ppo_critic-labeled.pth")
    else:
        _test(env=env, actor_model=args.actor_model)


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

    parser.add_argument('--mode', dest='mode', type=str, default='train')  # can be 'train' or 'test'
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='./tmp/ppo/temp_actor')  # your actor model filename
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='./tmp/ppo/temp_critic')  # your critic model filename

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    main(args)
