# References:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://keras.io

import gym
from keras import layers
from keras import models
from collections import deque
import random
import numpy as np
import tensorflow as tf
from statistics import mean
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import matplotlib.pyplot as plt

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class DQN:
    def __init__(self, num_outputs, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.model = models.Sequential()
        # self.model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=(110, 84, 4)))
        # self.model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=(210, 160, 4)))
        # self.model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
        # self.model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
        # self.model.add(layers.Flatten())
        # self.model.add(layers.Dense(512, activation='relu'))
        # self.model.add(layers.Dense(num_outputs, activation='linear'))

        # As per the paper
        #self.model.add(layers.Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=(110, 84, 4)))
        self.model.add(layers.Conv2D(16, (8, 8), strides=4, activation='relu', input_shape=(200, 300, 4)))
        
        self.model.add(layers.Conv2D(32, (4, 4), strides=2, activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dense(num_outputs, activation='linear'))

    def clone(self):
        return models.clone_model(self.model)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def predict(self, input, batch_size):
        return self.model.predict(input, batch_size=batch_size)

    def train(self, states, mask, target_Q_values):
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def save(self, name):
        self.model.save(name)

    def load(self, name):
        self.model = tf.keras.load_model(name)


class ReplayBuffer:
    def __init__(self, len):
        self.buffer = deque(maxlen=len)

    def append(self, *args):
        self.buffer.append(*args)

    def sample(self, size):
        x = random.sample(self.buffer, size)
        states, actions, rewards, next_states, dones = zip(*x)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


class StackedFrames:
    def __init__(self):
        self.frame_q = deque(maxlen=4)
        self.frame_q.append(np.zeros(INPUT_SHAPE))
        self.frame_q.append(np.zeros(INPUT_SHAPE))
        self.frame_q.append(np.zeros(INPUT_SHAPE))
        self.frame_q.append(np.zeros(INPUT_SHAPE))

    # As mentioned in the paper we first convert the RGB representation into 
    # gray scale and then down sample to 110 x 84. Since we do not have a 
    # limitation to make the images square we do not do it.
    def preprocess_frame(self, frame):
        # frame = np.squeeze(frame[:, :, 0])
        frame = tf.image.rgb_to_grayscale(frame)
        #frame = tf.keras.preprocessing.image.smart_resize(frame, (110, 84)) / 255.0
        frame = tf.keras.preprocessing.image.smart_resize(frame, INPUT_SHAPE) / 255.0
        
        frame = np.squeeze(frame)

        self.frame_q.append(frame)

        # Stack 4 frames as mentioned in the paper
        self.frames = np.dstack((self.frame_q[0], self.frame_q[1], self.frame_q[2], self.frame_q[3]))

        return

    def append(self, frame):
        return self.frame_q.append(frame)

    def get_frames(self):
        return self.frames


# state: current state
# returns the next action based on epsilon greedy policy
# As mentioned in the paper we use the epsilon greedy policy with epsilon annealed
# linearly from 1 to 0.1 and fixed at 0.1 thereafter. 
def select_next_action(env, network, episode, state):
    epsilon = max(1 - episode / EPS_ANNEALING_FACTOR, 0.1)
    if random.random() < epsilon:
        # Return random action
        return env.action_space.sample()
    else:
        # Select action based on the model
        Q = network.predict(state[np.newaxis], batch_size=1)
        Q_max = np.argmax(Q[0])

        return Q_max


def take_action(env, network, episode, sf):
    state = sf.get_frames()
    action = select_next_action(env, network, episode, state)
    next_state, reward, done, info = env.step(action)
    next_state = env.render(mode='rgb_array')
    next_sf = sf
    next_sf.preprocess_frame(next_state)

    # Populate the replay buffer so that we can sample batches from it
    replay_buffer.append((sf, action, reward, next_sf, done))

    return next_sf, reward, done, info


def train():
    # Sample batches from the replay buffer
    sfs, actions, rewards, next_sfs, dones = replay_buffer.sample(BATCH_SIZE)

    states = np.array([sfs[i].get_frames() for i in range(BATCH_SIZE)])
    next_states = np.array([next_sfs[i].get_frames() for i in range(BATCH_SIZE)])

    # Predict next Q values from the target model
    next_Q_values = target_network.predict(next_states, batch_size=BATCH_SIZE)

    # Select the optimal Q value for the next state by appying the Bellman Equation
    target_Q_values = rewards + (1 - dones) * GAMMA * np.max(next_Q_values, axis=1)

    # Crete a 1-hot encoding so that we select only the action that the agent has selected
    mask = tf.one_hot(actions, num_outputs)
    online_network.train(states, mask, target_Q_values)

    return


#ENV_NAME = 'Breakout-v0'
#ENV_NAME = 'BreakoutDeterministic-v4'
ENV_NAME = 'CartPole-v1'
MODEL_DIR = 'Models'
CHECKPOINT_STEP = 100

#INPUT_SHAPE = (210, 160)
#INPUT_SHAPE = (110, 84)
INPUT_SHAPE = (200, 300)

NUM_EPISODES = 2500
NUM_STEPS = 1000
BATCH_SIZE = 128
FRAME_PER_TRAIN = 8
TARGET_UPDATE_FRAMES = 500
REPLAY_BUFFER_LEN = 2500

EPS_ANNEALING_FACTOR = 1000
GAMMA = 0.99

# Setup the gym environment
env = gym.make(ENV_NAME)
env.reset()

#input_shape = env.observation_space.shape
num_outputs = env.action_space.n
optimizer = tf.keras.optimizers.RMSprop()
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Learning network
online_network = DQN(num_outputs,
                     loss_fn=tf.keras.losses.mean_squared_error,
                     optimizer=optimizer)

# target network
target_network = online_network.clone()
target_network.set_weights(online_network.get_weights())

# Replay Buffer
replay_buffer = ReplayBuffer(len=REPLAY_BUFFER_LEN)

sf = StackedFrames()

frame_count = 0
rewards = []
rolling_period = 100
rolling_rewards = []

# Start the episode
for e in range(NUM_EPISODES):
    env.reset()
    state = env.render(mode='rgb_array')
    sf.preprocess_frame(state)

    episode_rewards = 0
    for s in range(NUM_STEPS):
        frame_count += 1

        # state = state[:, :, 0]
        # state, reward, done, info = take_action(env, online_network, e, state[:, :, np.newaxis])
        sf, reward, done, info = take_action(env, online_network, e, sf)

        episode_rewards += reward

        if len(replay_buffer) >= BATCH_SIZE:
            train()

        # Copy the online model weights to the target model after regular intervals
        if frame_count % TARGET_UPDATE_FRAMES == 0:
            target_network.set_weights(online_network.get_weights())

        env.render()
        if done:
            break


    rewards.append(episode_rewards)
    rolling_rewards.append(mean(rewards[max(-rolling_period, -(e+1)):]))
    print("episodes={}, steps={}, reward={}, rreward={:.2f}".format(
        e, s, rewards[-1], rolling_rewards[e]))

    if e and not e % CHECKPOINT_STEP:
        target_network.save(f'{MODEL_DIR}/target_network/cps/{e}')
        online_network.save(f'{MODEL_DIR}/online_network/cps/{e}')


online_network.save(MODEL_DIR+'/online_network')
target_network.save(MODEL_DIR+'/target_network')


episodes = [i for i in range(e+1)] 
plt.plot(episodes, rewards, label='Rewards')
plt.plot(episodes, rolling_rewards, label=f'{rolling_period} Episode MA')
plt.show()

env.close()