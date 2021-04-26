from pyro.nn import PyroSample, PyroModule
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pyro
from pyro.nn import PyroModule, PyroSample, pyro_method
import matplotlib.pyplot as plt
import logging
import pyro.distributions as dist
import copy
import random


class RandomSystem:
    def __init__(self, num_items, maxlen_action , *args, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.maxlen_action = maxlen_action

    def forward(self, *args, **kwargs):
        return None

    def recommend(self, batch, *args, **kwargs):
        batch_size = len(batch['click'])
        action = 2 + torch.cat([
            torch.randperm(self.num_items - 2).unsqueeze(0)
            for _ in range(batch_size)
        ])
        action = action[:, :(self.maxlen_action - 1)]
        return action

