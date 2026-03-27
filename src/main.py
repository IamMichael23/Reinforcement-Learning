import os
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrapper import apply_wrapper
from mario import PPOAgent

ENV_NAME = "SuperMarioBros-v0"
N_ENVS = 8          # parallel environments for faster data collection
N_STEPS = 256       # steps per env per rollout (total = 8×256 = 2048 per update)
TOTAL_UPDATES = 10000


def make_env():
    env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrapper(env)
    return env


envs = [make_env() for _ in range(N_ENVS)]

state_dim = (4, 84, 84)
action_dim = envs[0].action_space.n

agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, n_envs=N_ENVS, n_steps=N_STEPS)
start_update = 0
episode_count = 0
if os.path.exists("mario_ppo.pth"):
    start_update, episode_count = agent.load()
    print(f"==================Loaded checkpoint (update {start_update}, {episode_count} episodes)==================")

states = []
for env in envs:
    s, _ = env.reset()
    states.append(np.array(s))
states = np.array(states)

last_x = np.zeros(N_ENVS)
stuck_counters = np.zeros(N_ENVS, dtype=int)

last_saved_ep = 0

for update in range(start_update, start_update + TOTAL_UPDATES):
    for step in range(N_STEPS):
        actions, log_probs = agent.choose_action(states)

        next_states = []
        rewards = np.zeros(N_ENVS)
        dones = np.zeros(N_ENVS, dtype=bool)

        for i, (env, action) in enumerate(zip(envs, actions)):
            ns, r, done, truncated, info = env.step(int(action))
            ns = np.array(ns)

            # Penalize Mario for standing still (encourage forward progress)
            current_x = info.get("x_pos", 0)
            if current_x == last_x[i]:
                stuck_counters[i] += 1
            else:
                stuck_counters[i] = 0
                last_x[i] = current_x
            if stuck_counters[i] > 50:
                r -= 1

            if done or truncated:
                episode_count += 1
                s, _ = env.reset()
                ns = np.array(s)
                last_x[i] = 0
                stuck_counters[i] = 0

            next_states.append(ns)
            rewards[i] = r
            dones[i] = done or truncated

        next_states = np.array(next_states)
        agent.store_transition(states, next_states, actions, log_probs, rewards, dones)
        states = next_states

    p_loss, v_loss, e_loss = agent.learn()

    if (update + 1) % 20 == 0:
        print(f"Update {update} | Episodes: {episode_count} | "
              f"Policy: {p_loss:.4f} | Value: {v_loss:.4f} | Entropy: {e_loss:.4f}")

    if (update + 1) % 50 == 0:
        agent.save(update=update + 1, episode_count=episode_count)
        print(f"--- Checkpoint saved at update {update + 1} ---")

agent.save(update=update + 1, episode_count=episode_count)
for env in envs:
    env.close()
