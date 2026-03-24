# Super Mario Bros - Deep Q-Network Agent

A reinforcement learning agent that learns to play Super Mario Bros using a Deep Q-Network (DQN). Built with PyTorch and OpenAI Gym.

---

## Architecture

```
Input (4x84x84 grayscale frames)
        |
  Conv2d(4, 32, 8x8, stride=4) + ReLU
        |
  Conv2d(32, 64, 4x4, stride=2) + ReLU
        |
  Conv2d(64, 64, 3x3, stride=1) + ReLU
        |
  Flatten (3136)
        |
  Linear(3136, 512) + ReLU
        |
  Linear(512, n_actions)
        |
  Q-values per action
```

## How It Works

The agent uses the **Bellman equation** to learn optimal actions:

```
Q(s, a) = r + γ * max_a' Q_target(s', a')
```

| Symbol | Meaning |
|--------|---------|
| `Q(s, a)` | Expected reward for taking action `a` in state `s` |
| `r` | Immediate reward |
| `γ` | Discount factor (how much future rewards matter) |
| `Q_target(s', a')` | Target network's estimate of future value |

**Key mechanisms:**
- **Epsilon-greedy exploration** — starts random, gradually exploits learned policy
- **Experience replay** — stores past experiences and samples random batches to break correlation
- **Target network** — frozen copy of the online network, synced periodically for stability
- **Frame stacking** — feeds 4 consecutive grayscale frames so the agent can perceive motion
- **Frame skipping** — repeats each action for 4 frames, reducing decision frequency

---

## Installation

```bash
# Clone the repository
git clone https://github.com/IamMichael23/Reinforcement-Learning.git
cd Reinforcement-Learning

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train the agent
```bash
source venv/bin/activate
python src/main.py
```

The agent automatically saves checkpoints to `mario_model.pth` and resumes training from the last checkpoint on restart.

### Watch the agent play
Set `render_mode='human'` in `src/main.py`:
```python
env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True, render_mode='human')
```

---

## Hyperparameters

| Parameter | Default | Effect of Increasing | Effect of Decreasing |
|-----------|---------|---------------------|---------------------|
| `exploration_rate_decay` | `0.999995` | Slower transition to exploitation — agent explores longer, sees more of the game but takes longer to use what it learned | Faster transition to exploitation — agent starts using learned policy sooner, but may miss strategies it hasn't discovered yet |
| `exploration_rate_min` | `0.1` | Agent always keeps a higher chance of random actions — helps discover new strategies but reduces consistency | Agent becomes more deterministic — better performance on learned behavior but can get permanently stuck on suboptimal strategies |
| `reward_decay` (γ) | `0.9` | Agent values future rewards more — better long-term planning (e.g., avoids short-term gains that lead to death) but slower to converge | Agent focuses on immediate rewards — learns faster initially but may miss strategies that require short-term sacrifice for long-term gain |
| `batch_size` | `32` | More stable gradient updates — smoother learning curve but slower per step and uses more memory | Noisier gradients — faster per step but learning can be unstable and oscillate |
| `lr` | `0.00025` | Larger weight updates — learns faster but risks overshooting optimal values, causing unstable Q-values or divergence | Smaller weight updates — more stable convergence but takes much longer to learn, may get stuck in local minima |
| `sync_every` | `10000` | Target network updates less frequently — more stable Q-targets but agent adapts slower to new experiences | Target network updates more often — adapts faster but can cause oscillating or diverging Q-values |
| `replay_buffer_size` | `100000` | Stores more diverse experiences — reduces correlation in training batches but uses more disk space and keeps stale experiences longer | Stores fewer experiences — agent forgets old experiences faster, more responsive to recent gameplay but less diverse training data |
| `skip` (frame skip) | `4` | Each action repeats for more frames — faster training (fewer decisions per episode) but less precise control, agent can't react quickly to threats | Each action repeats for fewer frames — more precise control but slower training and the agent must make many more decisions per episode |

### Reward Shaping

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stuck_threshold` | `50 steps` | Steps at same x-position before penalty kicks in |
| `stuck_penalty` | `-1` | Reward deducted per step while stuck |

---

## Project Structure

```
.
├── src/
│   ├── main.py            # Training loop and environment setup
│   ├── mario.py            # DQN agent (action selection, learning, save/load)
│   ├── neualNetwork.py     # CNN architecture
│   └── wrapper.py          # Frame skip, grayscale, resize, stack wrappers
├── mario_model.pth         # Saved model checkpoint
├── requirements.txt        # Python dependencies
└── README.md
```

## Action Sets

| Action Set | Actions | Training Speed |
|-----------|---------|---------------|
| `RIGHT_ONLY` (default) | NOOP, right, right+A, right+B, right+A+B | Fastest (5 actions) |
| `SIMPLE_MOVEMENT` | Adds left, left+A, down | Medium (7 actions) |
| `COMPLEX_MOVEMENT` | Full control including all combinations | Slowest (12 actions) |

---

## Technologies

- **PyTorch** — Neural network and optimization
- **OpenAI Gym** — Environment interface
- **gym-super-mario-bros** — Super Mario Bros environment
- **TorchRL** — Experience replay buffer
- **OpenCV** — Frame preprocessing
