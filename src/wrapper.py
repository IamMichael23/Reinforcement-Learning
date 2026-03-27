import numpy as np
from gym import Wrapper
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super(SkipFrame, self).__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        total_reward = np.clip(total_reward, -15, 15)
        return obs, total_reward, done, truncated, info
    

def apply_wrapper(env):
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4, lz4_compress=True)
    return env