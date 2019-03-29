import numpy as np


class BatchNormalizer:
    def __init__(self, momentum=1, epsilon=0.001):
        # momentum ration to keep how much of previous minibatches
        self._momentum = momentum
        self._previous_batches = []
        self._epsilon = epsilon

    def normalize_batch(self, batch):
        for i in range(len(batch)):
            if i == 0:
                states = batch[i][0]
                actions = batch[i][1]
                rewards = batch[i][2]
                next_states = batch[i][3]
                dones = batch[i][4]
            else:
                states = np.vstack((states, batch[i][0]))
                actions = np.vstack((actions, batch[i][1]))
                rewards = np.vstack((rewards, batch[i][2]))
                next_states = np.vstack((next_states, batch[i][3]))
                dones = np.vstack((dones, batch[i][4]))

        mean = np.mean(states, axis=0)

        var = np.var(states, axis=0)

        normalized_states = (states - mean) / np.sqrt(var + self._epsilon)

        mean = np.mean(next_states, axis=0)

        var = np.var(next_states, axis=0)

        normalized_next_states = (next_states - mean) / np.sqrt(var + self._epsilon)

        normalized_batch = [(normalized_states, actions, rewards, normalized_next_states, dones)]

        return normalized_batch


