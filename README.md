# Double Deep Q-Network for Mountain Car

DDQN implementation from scratch to solve the Mountain Car environment, with and without offline human gameplay data for replay buffer initialization.

## Problem

Mountain Car is a classic RL environment where a car is stuck in a valley and needs to reach the flag at the top of the right hill. The engine isn't strong enough to drive straight up — the car has to build momentum by swinging back and forth first.

The main challenge is sparse rewards: the agent gets -1 every step and only escapes when it reaches the goal. With pure random exploration, the agent almost never reaches the goal early in training, meaning almost no useful gradient signal for a long time.

## Approach

Implemented Double DQN (DDQN) from scratch using PyTorch, with two training variants:

- **DQN_offline_false** — standard online training, no prior data
- **DQN_offline_true** — replay buffer pre-filled with human gameplay data before online training begins

### Why Double DQN?
Standard DQN overestimates Q-values because it uses the same network for both action selection and evaluation. DDQN fixes this by decoupling them:
- Online network selects the best next action
- Target network evaluates that action's Q-value

### Why offline data?
Pre-filling the buffer with human demonstrations (including successful goal-reaching episodes) gives the agent useful gradient signal from episode 1 instead of waiting for random exploration to stumble onto the goal.

## Architecture

- 2 hidden layers, 64 neurons each, ReLU activations
- Replay buffer: 10,000 transitions
- Target network updated every 10 episodes
- Epsilon-greedy exploration, decayed exponentially
- Final model size: under 100 KB

## Results

Both models successfully solve Mountain Car. The offline-initialized variant converges faster due to early exposure to goal-reaching behavior. See `rewards_without_offline.png` for the reward curves.

## Files

| File | Description |
|------|-------------|
| `training.py` | DDQN training — online and offline variants |
| `testing.py` | Load final model and render one episode |
| `MountainCar_CollectOfflineData.py` | Collect human gameplay into car_dataset.csv |
| `MountainCar_PlayGame.py` | Play Mountain Car manually with keyboard |
| `CartPole_DQN_training.py` | Reference DQN on CartPole |
| `car_dataset.csv` | Human gameplay dataset (30+ episodes) |
| `DQN_offline_false.pth` | Trained model — no offline data |
| `DQN_offline_true.pth` | Trained model — offline initialized |
| `rewards_without_offline.png` | Reward curves comparing both runs |
| `Report.pdf` | Full implementation details and analysis |

## Tech Stack

Python, PyTorch, OpenAI Gymnasium, NumPy

## Usage

```bash
pip install gymnasium torch numpy pygame
pip install swig && pip install gymnasium[classic-control]

python training.py    # trains both models
python testing.py     # renders one episode with the trained model
```
