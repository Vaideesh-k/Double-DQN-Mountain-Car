import numpy as np
import gymnasium as gym
import pygame

import torch
import torch.nn as nn

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


def choose_action(model, state):

    inputs = np.array(
        [make_input(state, a) for a in range(NUM_ACTIONS)],
        dtype=np.float32
    )

    inp_t = torch.FloatTensor(inputs).to(DEVICE)

    with torch.no_grad():
        q_values = model(inp_t).squeeze().cpu().numpy()

    return int(np.argmax(q_values))


# model_state = torch.load(
#     'DQN_offline_false.pth',
#     map_location=DEVICE
# )

model_state = torch.load(
    'DQN_offline_true.pth',
    map_location=DEVICE
)

model = DQN_Arch1().to(DEVICE)

model.load_state_dict(model_state)

model.eval()


env = gym.make('MountainCar-v0', render_mode='human')

state, info = env.reset()

end_episode = False
total_reward = 0

while not end_episode:

    action = choose_action(model, state)

    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    state = np.array(next_state, dtype=np.float32)

    end_episode = terminated or truncated


print(f"Total reward: {total_reward}")

env.close()

pygame.display.quit()