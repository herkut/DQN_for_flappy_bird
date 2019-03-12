from ple.games.flappybird import FlappyBird
from ple import PLE
import numpy as np
import keras


class HyperBird:

    def __init__(self, input_shape=None, allowed_actions=None):
        self.input_shape = input_shape
        self.allowed_actions = allowed_actions
        self.epsilon = 0.3

    def pickAction(self, reward, state):
        # Explore
        if np.random.rand(1) < self.epsilon:
            print("Exploration")
            random_value = np.random.randint(2)
            return self.allowed_actions[random_value]
        # Exploit
        else:
            print("Exploitation")
        return self.allowed_actions[1]


############
############
def process_state(state):
    return np.array([ state.values() ])


def main():
    print(keras.backend.backend())

    game = FlappyBird()
    p = PLE(game, display_screen=True, state_preprocessor=process_state)
    agent = HyperBird(input_shape=p.getGameStateDims(), allowed_actions=p.getActionSet())

    REPLAY_MEMORY = []

    p.init()
    nb_frames = 10000
    reward = 0.0
    for i in range(nb_frames):
        if p.game_over():
            p.reset_game()

        state = p.getGameState()
        action = agent.pickAction(reward, state)
        reward = p.act(action)


if __name__ == "__main__":
    main()