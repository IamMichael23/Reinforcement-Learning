import sys
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mario import MarioAgent


class MarioLoadTests(TestCase):
    def test_load_corrupted_checkpoint_returns_false(self):
        agent = MarioAgent(state_dim=(4, 84, 84), action_dim=2)

        with patch("mario.torch.load", side_effect=RuntimeError("bad checkpoint")):
            loaded = agent.load(path="mario_model.pth")

        self.assertFalse(loaded)
