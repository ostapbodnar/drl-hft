import gym
import numpy as np
import plotly.express as px
import torch
import torch.nn.functional as Fun
from torch import nn

import gym_trading
from agent.ppo import PPO
from agent.nn_model import CnnLstmTwoHeadNN
from gym_trading.envs.base_environment import Observation


def plot_loss_curve(x, scores, figure_file=None):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])

    fig = px.line(x=x, y=running_avg, title='Average loss per one episode', labels={'x': 'Episodes', 'y': 'Loss'})
    if figure_file is None:
        fig.show()
    else:
        fig.write_image(figure_file)


if __name__ == '__main__':
    hyperparameters = {
        'timesteps_per_batch': 600_000,
        'max_timesteps_per_episode': 20_000,
        'gamma': 0.99,
        'n_updates_per_iteration': 10,
        'lr': 0.0001,
        'clip': 0.2,
        'render': False,
        'render_every_i': 10
    }
    config = dict(
        id=gym_trading.envs.HighFrequencyTrading.id,
        symbol='BTC_USDT',
        fitting_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_min_labeled.csv',
        testing_file='/mnt/c/Users/ostap/Desktop/diploma/kline_lob_btc_04_2021_val_min_labeled.csv',
        # testing_file='/Users/ostapbodnar/diploma_data/kline_lob_btc_04_2021_val_min_labeled.csv',
        max_position=20,
        window_size=100,
        seed=2,
        action_repeats=1,
        training=True,
        format_3d=False,
        reward_type='asymmetrical',
        ema_alpha=None,
        **hyperparameters
    )
    print(f"**********\n{config}\n**********")

    env: gym_trading.HighFrequencyTrading = gym.make(**config)

    model = PPO(policy_class=CnnLstmTwoHeadNN, env=env, **hyperparameters)
    obs = env.reset()

    t = 0

    model.actor.load_state_dict(torch.load('./ppo_actor-labeled.pth'))
    model.critic.load_state_dict(torch.load('./ppo_critic-labeled.pth'))
    print("weights loaded")

    loss_func = nn.MSELoss(reduction='sum')
    losses = []
    model.actor.eval()
    model.actor.train()

    while t < hyperparameters["timesteps_per_batch"]:
        score_history = []

        obs = model.env.reset()
        done = False
        ep_t = 0
        local_losses = []

        for ep_t in range(config["max_timesteps_per_episode"]):

            t += 1
            action_probs = model.get_action_probs(obs)
            target = env.labels[env.local_step_number]
            action = action_probs.flatten().argmax().item()
            obs, rew, done, _, _ = model.env.step(action)

            score_history.append(rew)

            A = torch.tensor([target], device=model.device)
            target_one_hot = Fun.one_hot(A, num_classes=env.action_space.n).to(model.device).float()

            loss = loss_func(action_probs, target_one_hot)

            model.actor_optim.zero_grad()
            loss.backward(retain_graph=True)
            model.actor_optim.step()
            local_losses.append(loss.item())

            if done:
                break

        losses.append(np.mean(local_losses))
        print(t, np.mean(local_losses))

        torch.save(model.actor.state_dict(), './ppo_actor-labeled.pth')
        torch.save(model.critic.state_dict(), './ppo_critic-labeled.pth')

        x = [i + 1 for i in range(len(losses))]
        plot_loss_curve(x, losses, 'plots/hft-losses.png')

    print(f"Max reward: {max(score_history)}\nMin reward: {min(score_history)}")
    print(f"Agent completed {env.broker.total_trade_count} trades")
    # Visualize results
    # env.plot_observation_history('plots/hft-plot-observation-history.png')
    env.plot_trade_history('plots/hft-plot-trade-history-labeled')
    env.reset()
