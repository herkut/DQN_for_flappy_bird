from datetime import datetime

import numpy as np

from ple.games.flappybird import FlappyBird
from ple import PLE

from hyper_bird import HyperBird


def process_state(state):
    s = np.zeros(len(state))
    i = 0
    for k, v in state.items():
        s[i] = v
        i += 1
    return s


EPISODES = 100000
FRAME_SIZE = 4  # Frame size is used to stack last n frame and q network is using this combined frames
BATCH_SIZE = 32


def train_the_bird():
    game = FlappyBird()
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    # input_shape = p.getGameStateDims()
    input_size = 8
    allowed_action = p.getActionSet()

    agent = HyperBird(batch_size=BATCH_SIZE, input_size=input_size, allowed_actions=allowed_action, frame_size=FRAME_SIZE)

    p.init()

    nb_frames = 10000
    reward = 0.0

    for e in range(EPISODES):
        p.reset_game()
        observation = p.getGameState()
        agent.set_init_state(observation)
        print("Episode: {}/{}".format(e, EPISODES))
        total_reward = 0
        for t in range(10000):
            # state = np.reshape(state, [1, input_size])
            action = agent.pick_action(observation, True)
            reward = p.act(allowed_action[action])
            next_observation = p.getGameState()

            # reward for passing a pipe
            if reward > 0:
                total_reward += 1
            # reward for staying alive
            elif reward == 0:
                total_reward += 0.001
            # punishment for dying
            elif reward < 0:
                total_reward = -1

            #agent.set_perception(next_observation, action, reward, p.game_over())

            agent.set_perception(next_observation, action, total_reward, p.game_over())

            if p.game_over():
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
        # state = np.array(list(state.items), np.float)
        action = agent.pick_action(state)
        reward = p.act(allowed_action[action])
        next_state = p.getGameState()
        # next_state = np.reshape(next_state, [1, input_shape[1]])

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
