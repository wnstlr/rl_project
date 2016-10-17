import numpy as np
from collections import deque
import random

class ReplayBuffer():
    def __init__(self, size):
        self.queue = deque()
        self.numel = 0
        self.max_size = size

    def add(self, s1, a, r, s2, t):
        trans = (s1, a, r, s2, t)
        if self.curr < self.size:
            self.queue.append(trans)
            self.numel += 1
        else:
            self.buffer.popleft()
            self.buffer.append(trans)

    def sample(self, n):
        if n < self.max_size:
            select = random.sample(self.buffer, n)
        else:
            select = random.sample(self.buffer, self.numel)

        return select

    def get_numel(self):
        return self.numel

    def get_size(self):
        return max_size

    def clear(self):
        self.buffer.clear()
        self.numel = 0
