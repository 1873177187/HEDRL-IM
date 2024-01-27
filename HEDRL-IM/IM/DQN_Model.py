
import torch.nn as nn

import torch.nn.functional as F


# Using a two-layer neural network as the DQN network #
class DQN(nn.Module):

    def __init__(self, dim):
        super(DQN, self).__init__()
        l1 = 64
        self.w1 = nn.Linear(int(dim), l1)
        self.w2 = nn.Linear(l1, 1)

    def forward(self, input):
        out = self.w1(input)
        out = F.relu_(out)

        out = self.w2(out)

        return out

