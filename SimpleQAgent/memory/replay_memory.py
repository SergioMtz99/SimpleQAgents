import numpy as np
from collections import deque
import random

class ReplayMemory():
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.memory = deque(maxlen = self.mem_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype = np.float32)
        actions = np.array(actions, dtype = np.int32)
        rewards = np.array(rewards, dtype = np.float32)
        next_states = np.array(next_states, dtype = np.float32)
        dones = np.array(dones, dtype = np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
