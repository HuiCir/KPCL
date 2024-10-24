from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn.functional as F


class BTModel(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.7):
        super(BTModel, self).__init__()
        self.tau: float = tau

        self.reward1 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )
        self.reward2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )


    def self_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)

    def btloss(self, z1: torch.Tensor, z2: torch.Tensor, ):
        f = lambda x: torch.exp(x / self.tau)
        preference = f(self.self_sim(z1, z2))
        rand_item = torch.randperm(z1.shape[0])
        nega = f(self.self_sim(z1, z2[rand_item])) + f(self.self_sim(z2, z1[rand_item]))

        return -torch.log(preference / (preference + preference + nega))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.reward1(z1)
        h2 = self.reward2(z2)
        loss = self.btloss(h1, h2).mean()
        return loss



