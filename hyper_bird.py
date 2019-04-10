import random
from collections import deque

import numpy as np

from keras import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import keras

import tensorflow as tf

from batch_normalizer import BatchNormalizer


class PortableBrain:
    def __init__(self, input_size, action_size, frame_size):
        self.input_size = input_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.00001
        self.observation_time = 1000

        self.time_step = 0

        self.update_time = 500

        self.frame_size = frame_size
        self.current_state = np.zeros(self.frame_size * input_size)
        self.next_state = np.zeros(self.frame_size * input_size)

        self.model = self.build_model_for_brain()
        self.target_model = self.build_model_for_brain()
        self.update_target_model()

    """
        Update the brain model for the agent according to the requirements
    """

    def build_model_for_brain(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.input_size*self.frame_size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self.mse_dqn, optimizer=Adam(lr=self.learning_rate))

        return model

    """
        Update loss function used in the brain model according to requirements
    """

    def huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
    """
        Mean squared error in DQN 1/N*(y-Q_t)^2 where y = r + (1-done) * gamma * Q_t1
    """
    def mse_dqn(self, y_true, y_pred):
        error = y_pred - y_true
        return K.mean(K.square(error))

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class HyperBird:

    def __init__(self, batch_size=32, input_size=None, allowed_actions=None, frame_size=4):
        self.input_size = input_size
        self.allowed_actions = allowed_actions
        self.batch_normalizer = BatchNormalizer()
        self.action_size = len(self.allowed_actions)
        self.batch_size = batch_size
        self.brain = PortableBrain(self.input_size, len(self.allowed_actions), frame_size)

    def pick_action(self, observation, is_being_trained=False):
        if is_being_trained:
            if np.random.rand() <= self.brain.epsilon:
                return random.randrange(self.action_size)

            act_values = self.predict(observation)

            return np.argmax(act_values[0])  # returns action
        else:
            act_values = self.predict(observation)

            return np.argmax(act_values[0])  # returns action

    def train_q_network(self):
        minibatch = random.sample(self.brain.memory, self.batch_size)
        normalized_minibatch = self.batch_normalizer.normalize_batch(minibatch)

        for states, actions, rewards, next_states, dones in normalized_minibatch:
            for i in range(np.shape(states)[0]):
                state_t = np.reshape(states[i], [1, len(states[i])])
                action_t = actions[i]
                reward_t = rewards[i]
                state_t1 = np.reshape(next_states[i], [1, len(next_states[i])])
                done = dones[i]

                y_t = self.brain.model.predict(state_t)

                q_values_t1 = self.brain.target_model.predict(state_t1)

                y_t[0][action_t] = reward_t + (1 - done) * self.brain.gamma * np.amax(q_values_t1[0])

                # train network
                self.brain.model.fit(state_t, y_t, epochs=1, verbose=0)
        if self.brain.epsilon > self.brain.epsilon_min:
            self.brain.epsilon *= self.brain.epsilon_decay

    """
    def remember(self, state, action, reward, next_state, done):
        self.brain.remember(state, action, reward, next_state, done)
    """

    def predict(self, observation):
        tmp_s = np.append(self.brain.current_state, observation)
        tmp_s = np.delete(tmp_s, np.s_[0:self.input_size])
        tmp_s = np.reshape(tmp_s, [1, len(tmp_s)])
        return self.brain.model.predict(tmp_s)

    def set_init_state(self, observation):
        for i in range(self.brain.frame_size):
            self.brain.current_state = np.append(self.brain.current_state, observation)
            self.brain.current_state = np.delete(self.brain.current_state,  np.s_[0:self.input_size])

    def set_perception(self, observation, action, reward, done):
        new_state = np.append(self.brain.current_state, observation)
        new_state = np.delete(new_state, np.s_[0:self.input_size])

        self.brain.remember(self.brain.current_state, action, reward, new_state, done)

        if self.brain.time_step > self.brain.observation_time:
            self.train_q_network()

        if self.brain.time_step % self.brain.update_time == 0:
            print('Updating the brain')
            self.update_brain()

        self.brain.current_state = new_state
        if done:
            print("Timestep: {}, reward: {}, epsilon: {:.2}".format(self.brain.time_step, reward, self.brain.epsilon))

        self.brain.time_step += 1

    def update_brain(self):
        self.brain.update_target_model()

    def load_brain_state(self, name):
        self.brain.model.load_weights(name)

    def save_brain_state(self, name):
        self.brain.model.save_weights(name)

