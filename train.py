import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import json

from model import Model

with open('config.json', 'r') as f:
    config = json.load(f)

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5", frameskip=1)
env = AtariPreprocessing(env)
env = FrameStackObservation(env, 4) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
target_model = Model().to(device)

optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], clipnorm=1.0) # Added clipnorm for gradient clipping
loss_function = nn.HuberLoss()

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
epsilon = config["epsilon"]

# Training Loop
while True:
    observation, _ = env.reset(seed=config["seed"])
    state = np.array(observation)
    episode_reward = 0

    for t in range(1, config["max_steps_per_episode"]):
        frame_count += 1

        if frame_count < config["epsilon_random_frames"] or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(config["num_actions"])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            state_tensor = state_tensor.unsqueeze(0)

            model.eval()
            with torch.no_grad():
                action_probs = model(state_tensor)

            action = torch.argmax(action_probs[0]).item()

        epsilon = max(config["epsilon_min"], config["epsilon_max"] - (config["epsilon_max"] - config["epsilon_min"]) * (frame_count / config["epsilon_greedy_frames"]))

        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        if frame_count % config["update_after_actions"] == 0 and len(done_history) > config["batch_size"]:
            model.train()

            indices = np.random.choice(range(len(done_history)), size=config["batch_size"])

            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])

            state_sample = torch.tensor(state_sample, dtype=torch.float32).to(device)
            state_next_sample = torch.tensor(state_next_sample, dtype=torch.float32).to(device)

            rewards_sample = [rewards_history[i] for i in indices]
            rewards_sample = torch.tensor(rewards_sample, dtype=torch.float32).to(device)

            action_sample = [action_history[i] for i in indices]
            action_sample = torch.tensor(action_sample, dtype=torch.long).to(device)

            done_sample = torch.tensor([float(done_history[i]) for i in indices]).to(device)

            future_rewards = target_model(state_next_sample)

            updated_q_values = rewards_sample + (1 - done_sample) * config["gamma"] * torch.amax(future_rewards, dim=1)

            masks = F.one_hot(action_sample, config["num_actions"])

            q_values = model(state_sample)
            q_action = torch.sum(torch.multiply(q_values, masks), dim=1)
            loss = loss_function(updated_q_values, q_action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if frame_count % config["update_target_network"] == 0:
            target_model.load_state_dict(model.state_dict())
            print(f"Running reward {running_reward} at episode {episode_count}, frame count {frame_count}")

            torch.save(model.state_dict(), 'model.pth')
            torch.save(target_model.state_dict(), 'target_model.pth')

        # test deque next
        if len(rewards_history) > config["max_memory_length"]:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:
        print(f"Solved at episode {episode_count}")
        break

    if (
        config["max_episodes"] > 0 and episode_count >= config["max_episodes"]
    ):  # Maximum number of episodes reached
        print("Stopped at episode {}!".format(episode_count))
        break

env.close()