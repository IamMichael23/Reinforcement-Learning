import os
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from wrapper import apply_wrapper
from mario import MarioAgent

ENV_NAME = 'SuperMarioBros-v0'
Number_OF_EPISODES = 50000

env = gym_super_mario_bros.make(ENV_NAME, apply_api_compatibility=True, render_mode=None)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrapper(env)

state_dim = (4, 84, 84)
action_dim = env.action_space.n

agent = MarioAgent(state_dim=state_dim, action_dim=action_dim)
if os.path.exists("mario_model.pth"):
    agent.load()
    agent.exploration_rate = 1
    print("==================Loaded checkpoint==================")

for episode in range(Number_OF_EPISODES):
    state, _ = env.reset()
    done = False
    truncated = False
    last_x = 0
    stuck_counter = 0
    while not done and not truncated:
        action = agent.choose_action(state)
        new_state, reward, done, truncated, info = env.step(action)

        # Penalize Mario for staying at the same x position
        current_x = info.get("x_pos", 0)
        if current_x == last_x:
            stuck_counter += 1
        else:
            stuck_counter = 0
            last_x = current_x

        if stuck_counter > 50:
            reward -= 1
            #print(f"Stuck at x={current_x} for {stuck_counter} steps, reward=-1")

        agent.store_replay(state, action, reward, next_state=new_state, done=done)
        agent.learn()
        state = new_state

    # Save checkpoint every 100 episodes
    if episode % 5 == 0:
        agent.save()
        print(f"Episode {episode} - Exploration Rate: {agent.exploration_rate:.4f} - Steps: {agent.step_counter}")

agent.save()
env.close()
   