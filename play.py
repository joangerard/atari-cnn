import numpy as np
import gym
import tensorflow as tf
from tensorflow.contrib.layers import flatten, conv2d, fully_connected
from collections import deque, Counter
import random
from datetime import datetime
from tensorflow.python.framework.errors_impl import NotFoundError

from dqn import DQN
from utils import preprocess_observation

if __name__ == '__main__':
    env = gym.make("Breakout-v0")
    n_outputs = env.action_space.n
    X_shape = (None, 88, 80, 1)
    X = tf.placeholder(tf.float32, shape=X_shape)
    reward_sum = 0
    with tf.Session() as sess:
        X_action = tf.placeholder(tf.int32, shape=(None,))
        mainDQN = DQN(sess, X, X_action, n_outputs, name="main")
        targetDQN = DQN(sess, X, X_action, n_outputs, name="target")
        episode = 998
        for i in range(9):
            done = False
            obs = env.reset()
            try:
                mainDQN.restore(episode)
                targetDQN.restore(episode)
                # self.tempDQN.restore(episode)
            except NotFoundError:
                print("save file not found")

            while not done:
                env.render()

                # get the preprocessed game screen
                obs = preprocess_observation(obs)
                # feed the game screen and get the Q values for each action
                actions = mainDQN.get_actions(obs)

                # get the action

                p = np.random.random(1).squeeze()
                action = np.argmax(actions, axis=-1) if p > 0.1 else np.random.randint(n_outputs)
                print("Action to play: ", action)

                # now perform the action and move to the next state, next_obs, receive reward
                next_obs, reward, done, _ = env.step(action)
                reward_sum += reward
                obs = next_obs
