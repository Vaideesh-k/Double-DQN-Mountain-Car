import numpy as np
import pandas as pd
import gymnasium as gym

import random
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

NUM_ACTIONS = 3
STATE_DIM = 2
INPUT_DIM = STATE_DIM + NUM_ACTIONS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DQN_Arch1(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def one_hot_action(action, n=NUM_ACTIONS):
    oh = np.zeros(n, dtype=np.float32)
    oh[action] = 1.0
    return oh


def make_input(state, action):
    return np.concatenate([
        np.array(state, dtype=np.float32),
        one_hot_action(action)
    ])


def get_all_q_values(model, state):
    inputs = np.array(
        [make_input(state, a) for a in range(NUM_ACTIONS)],
        dtype=np.float32
    )

    inp_t = torch.FloatTensor(inputs).to(DEVICE)

    with torch.no_grad():
        return model(inp_t).squeeze().cpu().numpy()


def boltzmann_action(model, state, epsilon, D=1.0):
    q_vals = get_all_q_values(model, state)

    q_bar = q_vals / (D + np.sum(np.abs(q_vals)) + 1e-8)

    exp_q = np.exp(q_bar - np.max(q_bar))
    probs = exp_q / (np.sum(exp_q) + 1e-8)

    if random.random() < epsilon:
        return int(np.random.choice(NUM_ACTIONS, p=probs))
    else:
        return int(np.argmax(q_bar))


def mc_physics(pos, vel, action):
    new_vel = vel + 0.001 * (action - 1) - 0.0025 * np.cos(3.0 * pos)
    new_vel = np.clip(new_vel, -0.07, 0.07)

    new_pos = pos + new_vel

    if new_pos <= -1.2 and new_vel < 0:
        new_vel = 0.0

    new_pos = np.clip(new_pos, -1.2, 0.6)

    terminated = bool(new_pos >= 0.45 and new_vel >= 0.0)

    return np.array([new_pos, new_vel], dtype=np.float32), -1.0, terminated


def generate_extra_transitions(state, action, reward, next_state, terminated):
    transitions = []

    pos, vel = float(state[0]), float(state[1])

    for a in range(NUM_ACTIONS):
        if a == action:
            transitions.append((state, a, reward, next_state, terminated))
        else:
            ns, r, term = mc_physics(pos, vel, a)
            transitions.append((state, a, r, ns, term))

    return transitions


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, terminated):
        self.buf.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(terminated),
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))

        s, a, r, ns, t = zip(*batch)

        return (
            np.array(s),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns),
            np.array(t, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buf)


def load_offline_data(path, min_score):
    state_data = []
    action_data = []
    reward_data = []
    next_state_data = []
    terminated_data = []

    dataset = pd.read_csv(path)

    dataset_group = dataset.groupby('Episode #')

    for play_no, df in dataset_group:

        start_idx = 0

        if isinstance(df.iloc[0, 1], str) and '{}' in df.iloc[0, 1]:
            start_idx = 1

        df = df[start_idx:]

        state = []

        for s in df.iloc[:, 1]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                state.append([float(val.strip(',')) for val in s])
            else:
                state.append(s)

        state = np.array(state)

        action = np.array(df.iloc[:, 2]).astype(int)
        reward = np.array(df.iloc[:, 3]).astype(np.float32)

        next_state = []

        for s in df.iloc[:, 4]:
            if isinstance(s, str):
                s = s.replace('[', '').replace(']', '').split()
                next_state.append([float(val.strip(',')) for val in s])
            else:
                next_state.append(s)

        next_state = np.array(next_state)

        terminated = np.array(df.iloc[:, 5]).astype(int)

        total_reward = np.sum(reward)

        if total_reward >= min_score:
            state_data.append(state)
            action_data.append(action)
            reward_data.append(reward)
            next_state_data.append(next_state)
            terminated_data.append(terminated)

    if not state_data:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([])
        )

    state_data = np.concatenate(state_data)
    action_data = np.concatenate(action_data)
    reward_data = np.concatenate(reward_data)
    next_state_data = np.concatenate(next_state_data)
    terminated_data = np.concatenate(terminated_data)

    return (
        state_data,
        action_data,
        reward_data,
        next_state_data,
        terminated_data
    )


def plot_reward(total_reward_per_episode, window_length):

    rewards = np.array(total_reward_per_episode)
    episodes = np.arange(1, len(rewards) + 1)

    moving_avg = np.array([
        np.mean(rewards[max(0, i - window_length + 1): i + 1])
        for i in range(len(rewards))
    ])

    plt.figure(figsize=(14, 5))

    plt.plot(
        episodes,
        rewards,
        alpha=0.35,
        color='steelblue',
        linewidth=0.9,
        label='Total Reward per Episode'
    )

    plt.plot(
        episodes,
        moving_avg,
        color='crimson',
        linewidth=2,
        label=f'Moving Average (window = {window_length})'
    )

    plt.axhline(
        y=-200,
        color='black',
        linestyle='--',
        alpha=0.5,
        label='Worst case (-200)'
    )

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Double DQN Training – MountainCar-v0')

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def DQN_training(
        env,
        offline_data,
        use_offline_data,
        E=50,
        num_episodes=5000,
        gamma=0.99,
        batch_size=64,
        buffer_size=100_000,
        N_u=4,
        N_T=200,
        hidden_dim=64,
        lr=1e-3,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        save_every=500):

    predict_net = DQN_Arch1(INPUT_DIM, hidden_dim).to(DEVICE)

    target_net = DQN_Arch1(INPUT_DIM, hidden_dim).to(DEVICE)

    target_net.load_state_dict(predict_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(predict_net.parameters(), lr=lr)

    loss_fn = nn.MSELoss()

    replay_buffer = ReplayBuffer(capacity=buffer_size)

    if use_offline_data:

        s_d, a_d, r_d, ns_d, t_d = offline_data

        if len(s_d) > 0:

            for i in range(len(s_d)):
                replay_buffer.push(
                    s_d[i],
                    int(a_d[i]),
                    float(r_d[i]),
                    ns_d[i],
                    bool(t_d[i])
                )

            print(f"[Offline] Buffer pre-loaded: {len(replay_buffer)} transitions")

        else:
            print("[Offline] WARNING: offline dataset is empty.")

    total_reward_per_episode = []

    epsilon = eps_start
    counter = 0

    for episode in range(num_episodes):

        state, _ = env.reset()

        state = np.array(state, dtype=np.float32)

        ep_reward = 0.0

        done = False
        truncated = False

        while not (done or truncated):

            action = boltzmann_action(predict_net, state, epsilon)

            next_state, reward, done, truncated, _ = env.step(action)

            next_state = np.array(next_state, dtype=np.float32)

            terminated = done

            if not (use_offline_data and episode < E):

                for trans in generate_extra_transitions(
                        state,
                        action,
                        reward,
                        next_state,
                        terminated):

                    replay_buffer.push(*trans)

            if counter % N_u == 0 and len(replay_buffer) >= batch_size:

                s_b, a_b, r_b, ns_b, t_b = replay_buffer.sample(batch_size)

                B = len(s_b)

                sel_inputs = np.array(
                    [make_input(ns_b[i], a)
                     for i in range(B)
                     for a in range(NUM_ACTIONS)],
                    dtype=np.float32
                )

                sel_t = torch.FloatTensor(sel_inputs).to(DEVICE)

                with torch.no_grad():

                    q_sel = predict_net(sel_t).squeeze()

                    q_sel = q_sel.view(B, NUM_ACTIONS)

                    best_actions = q_sel.argmax(dim=1).cpu().numpy()

                eval_inputs = np.array(
                    [make_input(ns_b[i], int(best_actions[i]))
                     for i in range(B)],
                    dtype=np.float32
                )

                eval_t = torch.FloatTensor(eval_inputs).to(DEVICE)

                with torch.no_grad():
                    q_tgt = target_net(eval_t).squeeze()

                r_t = torch.FloatTensor(r_b).to(DEVICE)

                term_t = torch.FloatTensor(t_b).to(DEVICE)

                targets = r_t + gamma * q_tgt * (1.0 - term_t)

                curr_inputs = np.array(
                    [make_input(s_b[i], int(a_b[i]))
                     for i in range(B)],
                    dtype=np.float32
                )

                curr_t = torch.FloatTensor(curr_inputs).to(DEVICE)

                predictions = predict_net(curr_t).squeeze()

                loss = loss_fn(predictions, targets.detach())

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            if counter % N_T == 0:
                target_net.load_state_dict(predict_net.state_dict())

            state = next_state

            ep_reward += reward

            counter += 1

        total_reward_per_episode.append(ep_reward)

        epsilon = max(eps_end, epsilon * eps_decay)

        if (episode + 1) % 50 == 0:

            avg100 = np.mean(total_reward_per_episode[-100:])

            print(
                f"Ep {episode+1:5d}/{num_episodes} | "
                f"Reward: {ep_reward:7.1f} | "
                f"Avg(100): {avg100:7.2f} | "
                f"ε: {epsilon:.4f} | "
                f"Buffer: {len(replay_buffer)}"
            )

        if (episode + 1) % save_every == 0:

            suffix = 'true' if use_offline_data else 'false'

            ckpt = f"DQN_offline_{suffix}_ep{episode+1}.pth"

            torch.save(predict_net.state_dict(), ckpt)

            print(f"Checkpoint saved: {ckpt}")

    return predict_net, np.array(total_reward_per_episode)


env = gym.make('MountainCar-v0')

path = 'car_dataset.csv'

min_score = -np.inf

offline_data = load_offline_data(path, min_score)

use_offline_data = True

final_model, total_reward_per_episode = DQN_training(
    env,
    offline_data,
    use_offline_data
)

model_name = (
    'DQN_offline_true'
    if use_offline_data
    else 'DQN_offline_false'
)

torch.save(final_model.state_dict(), f'{model_name}.pth')

print(f"Final model saved -> {model_name}.pth")

window_length = 50

plot_reward(total_reward_per_episode, window_length)

env.close()