# Super Mario Bros - Reinforcement Learning

A reinforcement learning project that trains agents to play Super Mario Bros. Two algorithms are implemented on separate branches:

| Branch | Algorithm | Description |
|--------|-----------|-------------|
| `main` | DQN (Deep Q-Network) | Value-based method using experience replay and target networks |
| `ppo` | PPO (Proximal Policy Optimization) | Policy gradient method with clipped surrogate objective |

---

## PPO Branch (`ppo`)

### Architecture

```
Input (4x84x84 grayscale frames)
        |
  ┌─────┴─────┐
  Actor        Critic
  (Policy)     (Value)
  |            |
  Conv2d → ReLU (shared architecture, separate weights)
  Conv2d → ReLU
  Conv2d → ReLU
  Flatten (3136)
  |            |
  FC(512)      FC(512)
  |            |
  Logits       Scalar
  (n_actions)  (state value)
```

### How PPO Works

PPO optimizes a clipped surrogate objective to prevent destructively large policy updates:

```
L_clip = min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A)
```

| Symbol | Meaning |
|--------|---------|
| `r(θ)` | Probability ratio between new and old policy |
| `A` | Advantage estimate (GAE) — how much better this action was than expected |
| `ε` | Clip range (0.2) — limits how far the policy can change per update |

**Key mechanisms:**
- **Generalized Advantage Estimation (GAE)** — balances bias vs variance in advantage estimates using λ=0.95
- **Reward normalization** — running standard deviation normalizer stabilizes critic training
- **Reward clipping** — caps frame-skipped rewards to (-15, 15) to prevent outliers
- **Parallel environments** — 8 envs collect data simultaneously for faster, more diverse rollouts
- **Clipped surrogate** — prevents catastrophic policy updates that destabilize training
- **Entropy bonus** — encourages exploration by penalizing overly confident policies
- **Frame stacking** — feeds 4 consecutive grayscale frames so the agent can perceive motion
- **Frame skipping** — repeats each action for 4 frames, reducing decision frequency

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_envs` | 8 | Parallel environments for data collection |
| `n_steps` | 256 | Steps per env per rollout (total 2048 per update) |
| `lr` | 2.5e-4 | Learning rate for both actor and critic |
| `gamma` | 0.99 | Discount factor for future rewards |
| `gae_lambda` | 0.95 | GAE smoothing parameter |
| `clip_epsilon` | 0.2 | PPO clipping range |
| `entropy_coeff` | 0.01 | Entropy bonus weight |
| `critic_coeff` | 0.5 | Value loss weight |
| `n_epochs` | 4 | Optimization passes per rollout |
| `batch_size` | 256 | Mini-batch size for gradient updates |

### Reward Shaping

| Parameter | Value | Description |
|-----------|-------|-------------|
| `stuck_threshold` | 50 steps | Steps at same x-position before penalty |
| `stuck_penalty` | -1 | Reward deducted per step while stuck |
| Reward clipping | (-15, 15) | Caps accumulated frame-skip rewards |

---

## DQN Branch (`main`)

See the `main` branch README for DQN-specific documentation.

---

## Installation

```bash
git clone https://github.com/IamMichael23/Reinforcement-Learning.git
cd Reinforcement-Learning

# Switch to PPO branch
git checkout ppo

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

The agent saves checkpoints to `mario_ppo.pth` every 50 updates and resumes from the last checkpoint on restart.

### Watch the agent play
Set `render_mode='human'` in `src/main.py`:
```python
env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True, render_mode='human')
```

---

## Project Structure

```
.
├── src/
│   ├── main.py            # Training loop and environment setup
│   ├── mario.py            # PPO agent (action selection, GAE, learning, save/load)
│   ├── neualNetwork.py     # Actor and Critic CNN architectures
│   └── wrapper.py          # Frame skip, grayscale, resize, stack wrappers
├── mario_ppo.pth           # Saved model checkpoint
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
- **OpenCV** — Frame preprocessing
