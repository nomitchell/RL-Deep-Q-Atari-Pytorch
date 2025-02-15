# DQN Atari Breakout in PyTorch

This repository implements a Deep Q-Network (DQN) agent in **PyTorch** to play Atari Breakout.

## Project Description

The agent learns to play directly from pixel input using the DQN algorithm.

## Files

  * `train.py`:  Main training script.
  * `model.py`:  Pytorch model architecture.
  * `test.py`:  Testing and visualization script.
  * `config.json`:  Hyperparameter configuration file.
  * `requirements.txt`: Python dependencies.
  * `readme.md`: This file.

## Dependencies

  * [Python 3.x](https://www.google.com/url?sa=E&source=gmail&q=https://www.python.org/)
  * [PyTorch](https://www.google.com/url?sa=E&source=gmail&q=https://pytorch.org/)
  * [Gymnasium](https://www.google.com/url?sa=E&source=gmail&q=https://gymnasium.farama.org/)
  * [NumPy](https://www.google.com/url?sa=E&source=gmail&q=https://numpy.org/)

Install dependencies using `pip`:

```bash
pip install -r requirements.txt # or install manually from list
```

## How to Run

1.  **Clone the repository & setup environment**

2.  **Train the agent:**

    ```bash
    python train.py
    ```

    Training progress (running reward, episode, frame count) will be printed to the console. Trained models are saved as `model.pth` and `target_model.pth`.

3.  **Test the trained agent:**

    ```bash
    python test.py
    ```

    This will load `model.pth` and visualize the agent playing Breakout.

## Configuration (`config.json`)

The `config.json` file contains training settings:

  * `learning_rate`: Optimization learning rate.
  * `gamma`: Discount factor.
  * `epsilon`: Exploration parameters (epsilon decay).
  * `batch_size`: Training batch size.
  * `max_memory_length`: Replay buffer size.
  * `update_target_network`: Target network update frequency.

Modify `config.json` to adjust hyperparameters.

## Model Architecture (`model.py`)

The model is a Convolutional Neural Network (CNN) in PyTorch, inspired by DQN research for Atari games. It processes stacked frames and outputs Q-values for each action.  See `model.py` for details on layers and structure.

## Based On

This implementation is based on https://arxiv.org/abs/1312.5602

-----