import torch
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
from neualNetwork import ActorNet, CriticNet


class RunningMeanStd:
    """Tracks running mean and variance for reward normalization."""
    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return x / (np.sqrt(self.var) + 1e-8)


class PPOAgent:
    """Proximal Policy Optimization agent for Mario."""
    def __init__(self, state_dim, action_dim, n_envs=8, n_steps=256,
                 lr=2.5e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coeff=0.01, critic_coeff=0.5,
                 n_epochs=4, batch_size=256):

        self.n_envs = n_envs
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.actor = ActorNet(state_dim, action_dim).to(self.device)
        self.critic = CriticNet(state_dim).to(self.device)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr)
        self.rollout = []
        self.reward_rms = RunningMeanStd()

    def _obs(self, states):
        """Convert raw pixels [0,255] → normalized tensor [0,1]."""
        t = torch.as_tensor(np.array(states), dtype=torch.float32) / 255.0
        if t.shape[-1] == 1:
            t = t.squeeze(-1)
        return t

    def choose_action(self, states):
        """Sample actions from the policy distribution (no gradient needed)."""
        with torch.no_grad():
            dist = Categorical(logits=self.actor(self._obs(states).to(self.device)))
        action = dist.sample()
        return action.cpu().numpy(), dist.log_prob(action).cpu()

    def store_transition(self, states, next_states, actions, log_probs, rewards, dones):
        self.rollout.append((
            self._obs(states),
            self._obs(next_states),
            torch.tensor(actions, dtype=torch.long),
            log_probs.detach(),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
        ))

    def learn(self):
        """PPO update: compute advantages, then optimize clipped surrogate."""
        # Stack rollout into tensors: shape (T=n_steps, N=n_envs, ...)
        obs, next_obs, actions, old_log_probs, rewards, dones = (
            torch.stack(x).to(self.device) for x in zip(*self.rollout)
        )
        self.rollout.clear()

        T, N = obs.shape[:2]  # T=timesteps, N=num_envs
        flat = lambda x: x.reshape(T * N, *x.shape[2:])

        # --- Reward normalization (divide by running std) ---
        self.reward_rms.update(rewards.cpu().numpy().flatten())
        rewards = rewards / (torch.tensor(float(self.reward_rms.var), dtype=torch.float32, device=self.device).sqrt() + 1e-8)

        # --- GAE (Generalized Advantage Estimation) ---
        # Estimates "how much better was this action than expected?"
        with torch.no_grad():
            values = self.critic(flat(obs)).reshape(T, N)
            next_values = self.critic(flat(next_obs)).reshape(T, N)

        adv = torch.zeros(T, N, device=self.device)
        gae = torch.zeros(N, device=self.device)
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * (~dones[t]) - values[t]  # TD error
            gae = delta + self.gamma * self.gae_lambda * (~dones[t]) * gae  # accumulate
            adv[t] = gae

        returns = (adv + values).reshape(-1)          # target for critic
        adv = ((adv - adv.mean()) / (adv.std() + 1e-8)).reshape(-1)  # normalize advantages
        flat_obs = flat(obs)
        actions = actions.reshape(-1)
        old_log_probs = old_log_probs.reshape(-1)

        # --- PPO clipped surrogate optimization ---
        p_losses, v_losses, e_losses = [], [], []
        for _ in range(self.n_epochs):
            for b in torch.randperm(len(flat_obs), device=self.device).split(self.batch_size):
                dist = Categorical(logits=self.actor(flat_obs[b]))

                # ratio = π_new(a|s) / π_old(a|s)
                ratio = (dist.log_prob(actions[b]) - old_log_probs[b]).exp()
                clipped = ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon)

                p_loss = -torch.min(ratio * adv[b], clipped * adv[b]).mean()  # policy loss
                v_loss = F.mse_loss(self.critic(flat_obs[b]), returns[b])      # value loss
                e_loss = dist.entropy().mean()                                 # entropy bonus

                # total = policy + 0.5·value - 0.01·entropy (entropy is maximized)
                self.optimizer.zero_grad()
                (p_loss + self.critic_coeff * v_loss - self.entropy_coeff * e_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.params, 0.5)
                self.optimizer.step()
                p_losses.append(p_loss.item())
                v_losses.append(v_loss.item())
                e_losses.append(e_loss.item())

        return np.mean(p_losses), np.mean(v_losses), np.mean(e_losses)

    def save(self, path="mario_ppo.pth", update=0, episode_count=0):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "update": update,
                    "episode_count": episode_count}, path)

    def load(self, path="mario_ppo.pth"):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        return ckpt.get("update", 0), ckpt.get("episode_count", 0)
