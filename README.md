# CartPole Reinforcement Learning Project

This project implements a reinforcement learning agent to solve the CartPole-v1 environment using Deep Q-Networks (DQN) with enhancements such as Prioritized Experience Replay (PER) and Double DQN (DDQN). The project also includes visualization tools for evaluating the agent's performance.

## Project Overview

- **CartPole Environment**: The CartPole-v1 environment is part of the OpenAI Gym library and serves as the environment for training the reinforcement learning agent.
- **Deep Q-Network (DQN)**: A neural network-based Q-learning algorithm is used to approximate the optimal Q-values for the CartPole environment.
- **Prioritized Experience Replay (PER)**: A technique used to prioritize important experiences during training.
- **Double DQN (DDQN)**: An enhancement to the standard DQN to reduce overestimation of Q-values.

## Project Structure

- **CartPole.py**: The main Python script that trains and evaluates the reinforcement learning agent.
- **PER.py**: Implements the Prioritized Experience Replay used in training.
- **DDQN_CartPole-v1_Dueling_PER_CNN.png**: A visualization showing the performance of the DDQN with PER after training.

## Installation

### Prerequisites

- **Python 3.x**: Ensure Python 3.x is installed on your machine.
- **Required Libraries**: Install the required libraries listed in `requirements.txt`.

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/CartPole.git
    cd CartPole
    ```

2. **Install Dependencies**:
    Install the necessary dependencies for running the project:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Train the DQN Agent**:
    Execute the main script to train the DQN agent on the CartPole-v1 environment:
    ```bash
    python CartPole.py
    ```

2. **Visualize the Results**:
    The script will output a performance visualization of the agent after training.

## Project Workflow

1. **Environment Setup**: Load the CartPole-v1 environment from OpenAI Gym.
2. **Model Training**: Train the reinforcement learning agent using DQN, with enhancements like PER and DDQN.
3. **Evaluation**: Evaluate the agent's performance and visualize the results.
