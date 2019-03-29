import random
from collections import deque
from time import sleep
from datetime import datetime

import numpy as np

from ple.games.flappybird import FlappyBird
from ple import PLE

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import keras

import tensorflow as tf


class UberBrain:
    def __init__(self, input_size, action_size):
        self._input_size = input_size
        self._action_size = action_size
        self._memory = deque(maxlen=1000)
        self._gamma = 0.95  # discount rate
        self._epsilon = 1.0  # exploration rate
        self._epsilon_min = 0.001
        self._epsilon_decay = 0.9999
        self._learning_rate = 0.001
        self._model = self.build_model_for_brain()
        self._target_model = self.build_model_for_brain()
        self.update_target_model()

    """
        Update the brain model for the agent according to the requirements
    """

    def build_model_for_brain(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self._input_size[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        model.compile(loss=self.huber_loss, optimizer=Adam(lr=self._learning_rate))

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

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    @property
    def model(self):
        return self._model

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def epsilon_min(self):
        return self._epsilon_min

    @property
    def target_model(self):
        return self._target_model

    @property
    def epsilon_decay(self):
        return self._epsilon_decay

    @property
    def gamma(self):
        return self._gamma

    @property
    def memory(self):
        return self._memory


class HyperBird:

    def __init__(self, input_shape=None, allowed_actions=None):
        self._input_shape = input_shape
        self._allowed_actions = allowed_actions
        self._action_size = len(self._allowed_actions)
        self._brain = UberBrain(self._input_shape, len(self._allowed_actions))

    def pick_action(self, state, is_being_trained=False):
        if is_being_trained:
            if np.random.rand() <= self._brain.epsilon:
                return random.randrange(self._action_size)

            act_values = self.predict(state)

            return np.argmax(act_values[0])  # returns action
        else:
            act_values = self.predict(state)

            return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self._brain.memory, batch_size)
        #print('Minibatch size: ' + str(len(minibatch)))
        for state, action, reward, next_state, done in minibatch:
            #print('Current reward:' + str(reward))
            target = self._brain.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self._brain.target_model.predict(next_state)[0]
                target[0][action] = reward + self._brain.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            # train network
            self._brain.model.fit(state, target, epochs=1, verbose=0)
        if self._brain.epsilon > self._brain.epsilon_min:
            self._brain.epsilon *= self._brain.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self._brain.remember(state, action, reward, next_state, done)

    def predict(self, state):
        return self._brain.model.predict(state)

    def update_brain(self):
        self._brain.update_target_model()

    def load_brain_state(self, name):
        self._brain.model.load_weights(name)

    def save_brain_state(self, name):
        self._brain.model.save_weights(name)


############
############
def process_state(state):
    return np.array([state.values()])


EPISODES = 100000


def train_the_bird():
    game = FlappyBird()
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    input_shape = p.getGameStateDims()
    allowed_action = p.getActionSet()
    agent = HyperBird(input_shape=input_shape, allowed_actions=allowed_action)

    batch_size = 32
    brain_update_frequency = 100

    p.init()

    nb_frames = 10000
    reward = 0.0

    for e in range(EPISODES):
        p.reset_game()

        for t in range(10000):
            #print('iteration: ' + str(t) + ' in episode: ' + str(e))
            state = p.getGameState()
            state = np.reshape(state, [1, input_shape[1]])
            action = agent.pick_action(state, True)
            reward = p.act(allowed_action[action])
            next_state = p.getGameState()
            next_state = np.reshape(next_state, [1, input_shape[1]])

            if reward > 0:
                reward = 1
            elif reward < 0:
                reward = -1

            agent.remember(state, action, reward, next_state, p.game_over())

            if len(agent._brain.memory) > batch_size:
                agent.replay(batch_size)

            if p.game_over():
                print("episode: {}/{}, score: {}, reward: {}, e: {:.2}".format(e, EPISODES, t, reward, agent._brain.epsilon))
                print('Updating the brain')
                agent.update_brain()
                break

    agent.save_brain_state('./brains/fist_trained_brain_3.h5')
    agent.save_brain_state('./brains/fist_trained_brain_' + datetime.datetime.now() + '.h5')
    return agent


def let_trained_bird_play_the_game(agent=None):
    game = FlappyBird()
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    input_shape = p.getGameStateDims()
    allowed_action = p.getActionSet()

    if agent is None:
        agent = HyperBird(input_shape=input_shape, allowed_actions=allowed_action)
        agent.load_brain_state('./brains/fist_trained_brain_2.h5')
    else:
        agent = agent

    batch_size = 32

    p.init()

    nb_frames = 10000
    reward = 0.0

    p.reset_game()

    for t in range(100000):
        state = p.getGameState()
        state = np.reshape(state, [1, input_shape[1]])
        action = agent.pick_action(state)
        reward = p.act(allowed_action[action])
        next_state = p.getGameState()
        next_state = np.reshape(next_state, [1, input_shape[1]])

        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        if p.game_over():
            break

    print("The uber bird played the game and gained " + str(reward) + " points")


def main():

    trained_bird = train_the_bird()

    let_trained_bird_play_the_game(trained_bird)


if __name__ == "__main__":
    main()
