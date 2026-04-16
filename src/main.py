import os
import gym
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrapper import apply_wrapper
from mario import PPOAgent
import time


class CompatEnv(gym.Wrapper):
    """Strips unsupported kwargs from reset()."""
    def reset(self, **kwargs):
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return self.env.reset(**kwargs)

WORLD1_LEVELS = [
    "SuperMarioBros-v0"
    # "SuperMarioBros-1-1-v0",
    # "SuperMarioBros-1-2-v0",
    # "SuperMarioBros-1-3-v0",
    # "SuperMarioBros-1-4-v0",
]
N_ENVS = 1        # 3 envs per level × 4 levels
N_STEPS = 1024      # steps per env per rollout (total = 12×1024 = 12288 per update)
TOTAL_UPDATES = 10000


def make_env(level):
    def _init():
        env = gym_super_mario_bros.make(level, apply_api_compatibility=True, render_mode="human")
        env = JoypadSpace(env, RIGHT_ONLY)
        env = apply_wrapper(env)
        env = CompatEnv(env)
        return env
    return _init


if __name__ == "__main__":
    # 3 envs per level = 12 total
    if N_ENVS > 1:
        env_fns = [make_env(level) for level in WORLD1_LEVELS for _ in range(3)]
    else:
        env_fns = [make_env(level) for level in WORLD1_LEVELS]
    envs = SubprocVecEnv(env_fns)

    state_dim = (4, 84, 84)
    action_dim = envs.action_space.n

    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, n_envs=N_ENVS, n_steps=N_STEPS)
    start_update = 0
    episode_count = 0
    if os.path.exists("mario_ppo.pth"):
        start_update, episode_count = agent.load()
        print(f"==================Loaded checkpoint (update {start_update}, {episode_count} episodes)==================")

    states = envs.reset()
    states = np.array(states)

    last_x = np.zeros(N_ENVS)
    stuck_counters = np.zeros(N_ENVS, dtype=int)

    # Reward tracking
    ep_rewards = np.zeros(N_ENVS)
    ep_max_x = np.zeros(N_ENVS)
    recent_rewards = []
    recent_max_x = []
    flag_count = 0

    for update in range(start_update, start_update + TOTAL_UPDATES):
        for step in range(N_STEPS):
            actions, log_probs = agent.choose_action(states)

            next_states, rewards, dones, infos = envs.step(actions)
            time.sleep(0.02)
            next_states = np.array(next_states)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=bool)

            for i, info in enumerate(infos):
                current_x = info.get("x_pos", 0)
                ep_rewards[i] += rewards[i]
                ep_max_x[i] = max(ep_max_x[i], current_x)

                if current_x == last_x[i]:
                    stuck_counters[i] += 1
                else:
                    stuck_counters[i] = 0
                    last_x[i] = current_x
                if stuck_counters[i] > 30:
                    rewards[i] -= 1
                    dones[i] = True

                if dones[i]:
                    episode_count += 1
                    recent_rewards.append(ep_rewards[i])
                    recent_max_x.append(ep_max_x[i])
                    if info.get("flag_get", False):
                        flag_count += 1
                        rewards[i] += info.get("time", 0) * 50.0
                    ep_rewards[i] = 0
                    ep_max_x[i] = 0
                    last_x[i] = 0
                    stuck_counters[i] = 0

            agent.store_transition(states, next_states, actions, log_probs, rewards, dones)
            states = next_states

        p_loss, v_loss, e_loss = agent.learn()

        if (update + 1) % 5 == 0:
            avg_rew = np.mean(recent_rewards) if recent_rewards else 0
            avg_x = np.mean(recent_max_x) if recent_max_x else 0
            print(f"Update {update} | Episodes: {episode_count} | "
                  f"Policy: {p_loss:.4f} | Value: {v_loss:.4f} | Entropy: {e_loss:.4f} | "
                  f"AvgReward: {avg_rew:.1f} | AvgMaxX: {avg_x:.0f} | Flags: {flag_count}")
            recent_rewards.clear()
            recent_max_x.clear()

        if (update + 1) % 10 == 0:
            agent.save(update=update + 1, episode_count=episode_count)
            print(f"--- Checkpoint saved at update {update + 1} ---")

    agent.save(update=update + 1, episode_count=episode_count)
    envs.close()
