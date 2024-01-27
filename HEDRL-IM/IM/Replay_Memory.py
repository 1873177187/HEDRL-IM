import random
from collections import namedtuple


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# 缓存池的数据结构 #
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity # 设置容量
        self.memory = [] # 设置List作为缓存池
        self.position = 0 # 用于记录最后数据存入的位置

    # 在容量未满的情况下存入四元组数据，如果容量超过上限，则覆盖之前的数据 #
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    # 以特定批次大小采用经验数据 #
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # 返回当前容量 #
    def __len__(self):
        return len(self.memory)
