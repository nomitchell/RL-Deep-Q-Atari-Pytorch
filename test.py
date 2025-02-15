import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torch
import numpy as np
from model import Model  # Assuming model.py contains your Model class

games = 5

def test_agent():
    """
    Tests a trained Deep Q-Network agent on Atari Breakout.
    Loads a model from 'model.pth' and runs it in the Breakout environment with human rendering.
    Includes debugging print statements to help identify issues.
    """

    print("Starting agent testing...")

    # 1. Initialize Environment
    env_name = "ALE/Breakout-v5"  # Or "BreakoutNoFrameskip-v4" if you trained on that
    try:
        env = gym.make(env_name, render_mode="human", frameskip=1)
        env = AtariPreprocessing(env)
        env = FrameStackObservation(env, 4)
        print(f"Successfully initialized environment: {env_name}")
    except Exception as e:
        print(f"Error initializing environment {env_name}: {e}")
        return

    # 2. Set Device (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Instantiate and Load Model
    try:
        model = Model().to(device)  # Create an instance of your Model class
        model.load_state_dict(torch.load("model.pth", map_location=device)) # Load saved weights
        model.eval()  # Set model to evaluation mode (important for inference)
        print("Successfully loaded model weights from 'model.pth'")
    except FileNotFoundError:
        print("Error: 'model.pth' not found. Make sure the model file is in the correct directory.")
        env.close()
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return

    # 4. Reset Environment and Initialize Variables
    observation, _ = env.reset()
    print("Initial observation shape:", np.array(observation).shape) # Check initial observation shape

    done = False
    total_reward = 0
    frame_count = 0

    # 5. Game Loop
    for _ in range(games):
        observation, _ = env.reset()
        done = False
        total_reward = 0
        frame_count = 0

        # may be needed in initial stages
        # observation, reward, done, _, _ = env.step(1)
        while not done:
            frame_count += 1

            # a. Preprocess State (Normalize and Convert to Tensor)
            state = np.array(observation, dtype=np.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # To tensor, add batch dim

            # b. Agent's Action Selection (Inference)
            with torch.no_grad(): # Disable gradient calculation for inference
                action_probs = model(state_tensor) # Forward pass through the model
                action = torch.argmax(action_probs, dim=1).item() # Choose action with highest Q-value

            # Debug Print Statements (Uncomment for detailed debugging)
            #print(f"Frame: {frame_count}, State shape: {state.shape}, Tensor shape: {state_tensor.shape}")
            #print(f"Action Probabilities: {action_probs}")
            #print(f"Selected Action: {action}")


            # c. Validate Action (Sanity Check - should rarely trigger if model output is correct)
            if action >= env.action_space.n:
                print(f"WARNING: Invalid action detected! Action: {action}, Action Space Size: {env.action_space.n}")
                print("This indicates a potential issue with your model's output layer or action selection.")
                break # Exit loop if invalid action to prevent errors

            # d. Take Action in Environment
            try:
                observation, reward, done, _, _ = env.step(action)
                total_reward += reward
                # print(f"Frame: {frame_count}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}, Done: {done}") # Optional reward tracking per frame

            except Exception as env_step_error:
                print(f"Error during env.step(action): {env_step_error}")
                print(f"Action that caused the error: {action}")
                break # Exit loop if environment step fails


            if done:
                print(f"\nEpisode finished after {frame_count} frames.")
                print(f"Total episode reward: {total_reward}")


    # 6. Close Environment
    env.close()
    print("Environment closed.")
    print("Testing complete.")

if __name__ == "__main__":
    test_agent()