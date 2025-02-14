import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

config = {
    "seed": 42,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.1,
    "epsilon_max": 1.0,
    "batch_size": 32,
    "max_steps_per_episode": 10000,
    "max_episodes": 10,
    "render_mode": "human",
    "num_actions": 4,
    "epsilon_random_frames": 50000,
    "epsilon_greedy_frames": 1000000.0,
    "max_memory_length": 100000,
    "update_after_actions": 4,
    "update_target_network": 10000

}

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", render_mode=config["render_mode"])
env = AtariPreprocessing(env)
env = FrameStackObservation(env, 4)

env.seed(config["seed"])

model = Model()
target_model = Model()

optimizer = optim.Adam(model.parameters(), lr=0.00025)
loss_function = nn.HuberLoss()

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Reset the environment to generate the first observation
while True:
    observation, _ = env.reset(seed=config["seed"])
    state = np.array(observation)
    episode_reward = 0

    for t in range(1, config["max_steps_per_episode"]):
        frame_count += 1

        if frame_count < config["epsilon_random_frames"] or config["epsilon"] > np.random.rand(1)[0]:
            action = np.random.choice(config["num_actions"])
        else:
            state_tensor = torch.tensor(state)
            state_tensor = ### left off here at expand

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()