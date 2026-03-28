from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


env = gym_super_mario_bros.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, COMPLEX_MOVEMENT)

done = True
truncated = False
for step in range(5000):
    if done or truncated:
        state = env.reset()
    state, reward, done, truncated, info = env.step(env.action_space.sample())
    env.render()

env.close()