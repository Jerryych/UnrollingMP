import torch
from torch import nn


class Instance_block(nn.Module):
    '''
    Update specific X[i] and phi
    '''

    def __init__(self, p, tr, step, phi_step, thresh):
        super(Instance_block, self).__init__()
        self.p = p
        self.training = tr
        self.step = nn.Parameter(step, requires_grad=self.training)
        self.phi_step = nn.Parameter(phi_step, requires_grad=self.training)
        self.thresh = torch.as_tensor(thresh)
        self.thresh_func = nn.Hardshrink(self.thresh)

    def forward(self, phi, X, Y, idx):
        # Update X[i]
        X_out = torch.zeros_like(X)
        X_out[:, idx] = X[:, idx].clone() - (-1) * self.step * torch.transpose(phi.clone(), 0, 1) @ (Y[:, idx].clone() - phi.clone() @ X[:, idx].clone())
        X_out[:, idx] = self.thresh_func(X_out[:, idx].clone())
        # Update X[0] ~ X[i - 1]
        for i in range(idx):
            X_out[:, i] = X[:, i].clone() - (-1) * self.step * torch.transpose(phi.clone(), 0, 1) @ (Y[:, i].clone() - phi.clone() @ X[:, i].clone())
            X_out[:, i] = self.thresh_func(X_out[:, i].clone())
        # Update phi with X[0] ~ X[i]
        #idx += 1
        #phi_out = torch.zeros_like(phi)
        #phi_out = phi.clone() - self.phi_step * 4 * phi.clone() @ (torch.transpose(phi.clone(), 0, 1) @ phi.clone() - torch.eye(self.p)) - 2 * (Y[:, : idx].clone() - phi.clone() @ X_out[:, : idx].clone()) @ torch.transpose(X_out[:, : idx].clone(), 0, 1)
        # To unit column vector
        #phi_out = phi_out.clone() / torch.norm(phi_out.clone(), dim=0)
        phi_out = phi.clone()

        return phi_out, X_out


class Constraint_block(nn.Module):
    '''
    Update X[0] ~ X[N], N: number of instance
    '''

    def __init__(self, N, p, tr, steps, phi_steps, threshs):
        super(Constraint_block, self).__init__()
        self.N = N
        self.p = p
        layers = [Instance_block(self.p, tr, steps[i], phi_steps[i], threshs[i]) for i in range(self.N)]
        self.inst_blocks = nn.ModuleList(layers)

    def forward(self, phi, X, Y):
        for idx in range(self.N):
            phi, X = self.inst_blocks[idx](phi, X, Y, idx)
        return phi, X
