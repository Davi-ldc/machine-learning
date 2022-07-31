import gym
gym.new_step_api = True

env = gym.make("Taxi-v3", render_mode="human")

env.reset()
env.render()