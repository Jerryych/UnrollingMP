import torch
from torch import nn


class Instance_block(nn.Module):
'''
Update specific X[i] and phi
'''

    def __init__(self, tr, step, phi_step, thresh):
        super(Instance_block, self).__init__()
        self.training = tr
        self.step = nn.Parameter(step, requires_grad=self.training)
        self.phi_step = nn.Parameter(phi_step, requires_grad=self.training)
        self.thresh = nn.Parameter(thresh, requires_grad=self.training)
        self.thresh_func = nn.Threshold(self.thresh, 0)

    def forward(self, phi, X, Y, idx):
        # Update X[i]
        X[:, idx] = X[:, idx] - (-1) * self.step * torch.transpose(phi, 0, 1) @ (Y[:, idx] - phi @ X[:, idx])
        X[:, idx] = self.thresh_func(X[:, idx])
        # Update X[0] ~ X[i - 1]
        for i in range(idx):
            X[:, i] = X[:, i] - (-1) * self.step * torch.transpose(phi, 0, 1) @ (Y[:, i] - phi @ X[:, i])
            x[:, i] = self.thresh_func(X[:, i])
        # Update phi with X[0] ~ X[i]
        idx += 1
        phi = phi - self.phi_step * 4 * phi @ (torch.transpose(phi, 0, 1) @ phi - torch.eye(p)) - 2 * (Y[:, : idx] - phi @ X[:, : idx]) @ torch.transpose(X[:, : idx], 0, 1)
        # To unit column vector
        phi = phi / torch.norm(phi, dim=0)

        return phi, X


class Constraint_block(nn.Module):
'''
Update X[0] ~ X[N], N: number of instance
'''

    def __init__(self, tr, steps, phi_steps, threshs):
        super(Constraint_block, self).__init__()
        self.N = N
        layers = [Instance_block(tr, steps[i], phi_steps[i], threshs[i]) for i in range(self.N)]
        self.inst_blocks = nn.ModuleList(layers)

    def forward(self, phi, X, Y):
        for idx in range(self.N):
            phi, X = self.inst_blocks[i](phi, X, Y, idx)
        return phi, X
