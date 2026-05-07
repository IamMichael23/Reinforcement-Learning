# Super Mario Bros — Reinforcement Learning Agent

A reinforcement learning agent that learns to play *Super Mario Bros* from raw 84×84 grayscale frames. Two algorithms in two branches: a DDQN baseline on `main`, and a PPO upgrade on `ppo`. Built with PyTorch, OpenAI Gym, and `gym-super-mario-bros`.

> _Add a training GIF here once you have one — a 5-10 second clip of the agent clearing 1-1 is the single highest-leverage thing you can put on this README._

## Branches

| Branch | Algorithm | Why it exists |
|---|---|---|
| `main` | DDQN (Double DQN with target network + replay buffer) | First implementation. Off-policy, replay-driven, epsilon-greedy. Functions as the baseline. |
| `ppo` | PPO (clipped surrogate, Actor-Critic, GAE) | Upgrade. On-policy, parallel envs, stable updates. Clears Mario World 1 reliably; ~30× the win rate of the DDQN baseline. |

```bash
git checkout main   # DDQN baseline
git checkout ppo    # PPO upgrade
```

## Headline result (PPO)

- **86% win rate** clearing levels in World 1
- **~30× improvement** over the DDQN baseline on the same evaluation
- Trained on raw 84×84×4 stacked grayscale frames (no hand-crafted features)
- 8 parallel environments, 256 steps per rollout, ~1024 batch size

## Algorithm — PPO (`ppo` branch)

The agent is an Actor-Critic CNN. Both heads share a 3-layer convolutional trunk (Conv8/4 → Conv4/2 → Conv3/1), then split:

- **Actor** outputs logits over actions; sampled via `torch.distributions.Categorical`
- **Critic** outputs a scalar state value used for advantage estimation

Training loop (`src/mario.py` on `ppo`):

1. Collect rollouts from `n_envs` parallel Mario instances using the current policy
2. Compute advantages with **GAE** (γ=0.99, λ=0.95)
3. Normalize rewards via a running mean/variance (`RunningMeanStd`)
4. Optimize the **clipped surrogate** loss for `n_epochs` over the rollout:
   - Policy loss: `min(ratio · A, clip(ratio, 1±ε) · A)` with ε=0.15
   - Value loss: MSE against returns, weighted 0.5
   - Entropy bonus: weighted 0.005, encourages exploration
5. Clip gradients at `‖g‖ = 0.5`, Adam at `lr=2e-4`

```
Rollout (T × N envs)
       │
       ▼
GAE advantages ──► clipped policy loss
       │                │
       │                ▼
       └──► critic loss + entropy bonus
                        │
                        ▼
                   Adam update
```

## Algorithm — DDQN (`main` branch)

The DDQN baseline lives on `main` for comparison. It's a single Q-network with a frozen target network synced every 10K steps, an LRU replay buffer of 100K transitions, and ε-greedy exploration decaying from 1.0 → 0.1.

| Mechanism | Detail |
|---|---|
| Network | 3 conv layers → FC(512) → Q-values per action |
| Update rule | DDQN target: action selection from online net, evaluation from target net |
| Replay | TorchRL `LazyMemmapStorage`, 100K capacity |
| Exploration | ε-greedy, decay 0.999995, floor 0.1 |
| Frame stacking | 4 grayscale 84×84 frames |
| Frame skip | 4 |

## Why PPO beats DDQN here

- DDQN's epsilon-greedy exploration gets stuck on tricky platforming sequences (gaps, enemies). PPO's stochastic policy explores more naturally.
- DDQN's replay buffer mixes old and new experience; on a non-stationary task like Mario, that hurts more than it helps.
- PPO's parallel env rollouts (8 envs × 256 steps) give a stronger gradient signal per update than DDQN's small replay batches.

## Run it locally

```bash
git clone https://github.com/IamMichael23/Super-Mario-With-PPO-RL.git
cd Super-Mario-With-PPO-RL

# Pick the algorithm
git checkout ppo                         # or `main` for DDQN

# Set up
python -m venv venv
source venv/bin/activate                  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train
python src/main.py
```

Checkpoints save to `mario_ppo.pth` (PPO) or `mario_model.pth` (DDQN). Training resumes from the latest checkpoint on restart.

## Stack

PyTorch · OpenAI Gym · `gym-super-mario-bros` · `nes-py` · `stable-baselines3` (SubprocVecEnv) · OpenCV · TorchRL

## File layout

```
src/
├── main.py            # Training loop, env setup, vec env (PPO uses SubprocVecEnv)
├── mario.py           # PPOAgent (ppo branch) / MarioAgent DDQN (main branch)
├── neualNetwork.py    # ActorNet + CriticNet (PPO) / DQN (main)
├── wrapper.py         # Frame skip, grayscale, resize, stack
└── gameSetUp.py       # Env init helpers
requirements.txt
```
