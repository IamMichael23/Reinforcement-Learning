import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrapper import apply_wrapper
from mario import MarioAgent

ENV_NAME = 'SuperMarioBros-v0'
Number_OF_EPISODES = 50000

env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True, render_mode='human')
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrapper(env)

state_dim = (4, 84, 84)
action_dim = env.action_space.n

agent = MarioAgent(state_dim=state_dim, action_dim=action_dim)

for episode in range(Number_OF_EPISODES):
    state, _ = env.reset()
    done = False    
    while not done:
        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)
        agent.store_replay(state, action, reward, next_state=new_state, done=done)
        agent.learn()
        state = new_state
env.close()
   
