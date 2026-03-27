import sys
from pathlib import Path
from unittest import TestCase

import torch


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mario import MarioAgent


class _FakeBuffer:
    def __init__(self, batch):
        self._batch = batch

    def __len__(self):
        return 999

    def sample(self, _):
        return self._batch


class _FakeLoss:
    def backward(self):
        return None


class _FakeOptimizer:
    def zero_grad(self):
        return None

    def step(self):
        return None


class _CallNet:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.device = "cpu"

    def __call__(self, _):
        return self._outputs.pop(0)


class MarioDDQNTests(TestCase):
    def test_learn_uses_online_argmax_and_target_value_for_next_q(self):
        agent = MarioAgent(state_dim=(4, 84, 84), action_dim=2)
        agent.batch_size = 2
        agent.reward_decay = 0.9
        agent.replay_buffer = _FakeBuffer(
            {
                "state": torch.zeros((2, 4, 84, 84), dtype=torch.float32),
                "action": torch.tensor([1, 0]),
                "reward": torch.tensor([1.0, 2.0]),
                "next_state": torch.zeros((2, 4, 84, 84), dtype=torch.float32),
                "done": torch.tensor([False, False]),
            }
        )
        agent.sync_target_network = lambda: None
        agent.optimizer = _FakeOptimizer()

        # First online call is Q(s, a), second is Q_online(s', a')
        agent.online_net = _CallNet(
            [
                torch.tensor([[7.0, 8.0], [9.0, 6.0]]),
                torch.tensor([[1.0, 3.0], [4.0, 2.0]]),
            ]
        )
        # Q_target(s', a')
        agent.target_net = _CallNet([torch.tensor([[10.0, 20.0], [30.0, 40.0]])])

        recorded = {}

        def capture_loss(pred, target):
            recorded["pred"] = pred.detach().clone()
            recorded["target"] = target.detach().clone()
            return _FakeLoss()

        agent.loss_fn = capture_loss
        agent.learn()

        # DDQN target:
        # online argmax actions at next state = [1, 0]
        # target values at those actions = [20, 30]
        # target = reward + gamma * next_q = [19, 29]
        expected_target = torch.tensor([19.0, 29.0])
        self.assertTrue(torch.allclose(recorded["target"], expected_target))
