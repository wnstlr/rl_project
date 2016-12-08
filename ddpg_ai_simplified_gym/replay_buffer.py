""" 
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size, gamma, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.gamma = gamma
        # random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
            
    def addMultiple(self, experiences):
        for experience in experiences:
            s = experience[0]
            a = experience[1]
            r = experience[2]
            t = experience[3]
            s2 = experience[4]
            self.add(s, a, r, t, s2)

    def addWithLabel(self, experiences):
        assert len(experiences) > 0
        terminal = experiences[len(experiences) - 1]
        experiences[len(experiences) - 1] = (terminal[0], terminal[1], terminal[2], terminal[2], terminal[4])
        for i in xrange(len(experiences) - 2, -1, -1):
            experience = experiences[i]
            reward = experience[2]
            target = (experiences[i+1])[3]
            total_reward = reward + self.gamma * target
            experiences[i] = (experience[0], experience[1], experience[2], total_reward, experience[4])
        self.addMultiple(experiences)
    
    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        # I am not sure whether minibatches from the replay buffer should be sampled with or without replacement
        # for DDPG (currently sampling with replacement as done in dqn-hfo implementation)
        
        ''' 
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        '''
        
        indices = np.random.randint(0, self.count, batch_size)
        for index in indices:
            batch.append(self.buffer[index])
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.deque.clear()
        self.count = 0


