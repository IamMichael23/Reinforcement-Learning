import torch
import numpy as np  
from neualNetwork import DQN
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage


class MarioAgent:    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.step_counter = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.999995
        self.exploration_rate_min = 0.1

        
        # Replay buffer
        self.replay_buffer = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))


        # Q-target network
        self.reward_decay = 0.9
        self.sync_every = 10000  # no. of experiences between Q_target & Q_online sync
        self.batch_size = 32
        self.lr = 0.00025
        
        
        # Mario's DNN to predict the most optimal action
        self.online_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim, freeze=True).to(self.device)
        
        # Optimization
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = torch.nn.MSELoss()

       
    def choose_action(self, state):
        """Given a state, choose an epsilon-greedy action"""
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_dim)
        # EXPLOIT
        state = torch.tensor(np.array(state), dtype=torch.float32).squeeze(-1).\
                             unsqueeze(0).to(self.online_net.device)
        q_values = self.online_net(state)   
        action = torch.argmax(q_values, dim=1).item()
        return action

    def decay_exploration(self):
        """Decay exploration rate"""
        self.exploration_rate = max(self.exploration_rate_min, 
                                    self.exploration_rate * self.exploration_rate_decay)
        
    def store_replay(self, state, action, reward, next_state, done):
        """Store experience to replay buffer"""
        self.replay_buffer.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32).squeeze(-1),
            "action": torch.tensor(action),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32).squeeze(-1),
            "done": torch.tensor(done)
        }, batch_size=[]))

    def sync_target_network(self):
        """Copy online network weights to target network"""
        if self.step_counter % self.sync_every == 0 and self.step_counter > 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def learn(self):
        """Update online network using DDQN Bellman target."""
        if len(self.replay_buffer) < self.batch_size:
            return

        self.sync_target_network()

        # Sample batch of experiences (s, a, r, s', done)
        batch = self.replay_buffer.sample(self.batch_size)
        states = batch["state"].to(self.online_net.device)         # s
        actions = batch["action"].to(self.online_net.device)       # a
        rewards = batch["reward"].to(self.online_net.device)       # r
        next_states = batch["next_state"].to(self.online_net.device)  # s'
        dones = batch["done"].to(self.online_net.device)

        # Q(s, a) - predicted Q values from online network
        pred_q_values = self.online_net(states)
        pred_q_values = pred_q_values[np.arange(self.batch_size), actions.squeeze()]

        # DDQN target:
        # choose next action from online network, evaluate it with target network
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q_values = rewards + (1 - dones.float()) * self.reward_decay * next_q_values

        # Minimize loss between predicted Q(s,a) and DDQN target
        loss = self.loss_fn(pred_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        self.decay_exploration()

    def save(self, path="mario_model.pth"):
        """Save model weights and training state to disk"""
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "exploration_rate": self.exploration_rate,
            "step_counter": self.step_counter
        }, path)

    def load(self, path="mario_model.pth"):
        """Load model weights and training state from disk"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            print(f"Checkpoint load failed ({path}): {error}. Starting from scratch.")
            return False

        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.exploration_rate = checkpoint["exploration_rate"]
        self.step_counter = checkpoint["step_counter"]
        return True
