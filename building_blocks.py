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

    def forward(self, phi, X, X_prv, Y, idx):
        # Combine update X[i] & update X[0] ~ X[i + 1] together in matrix form
        if idx == 0:
            # Update vector X[0] only, store X[0] in matrix X_out
            X_out = torch.zeros_like(X)
            x_tmp = X[:, idx].clone() - (-2) * self.step * torch.transpose(phi, 0, 1) @ (Y[:, idx].clone() - phi @ X[:, idx].clone())
            X_out[:, idx] = self.thresh_func(x_tmp)
        else:
            # Update X[0] ~ X[i] in matrix form
            X_out = torch.zeros_like(X)
            x_tmp = X[:, : idx + 1].clone() - (-2) * self.step * torch.transpose(phi, 0, 1) @ (Y[:, : idx + 1].clone() - phi @ X[:, : idx + 1].clone())
            X_out[:, : idx + 1] = self.thresh_func(x_tmp)
        # Update phi with X[0] ~ X[i]
        idx += 1
        phi_out = phi - self.phi_step * (4 * phi @ (torch.transpose(phi, 0, 1) @ phi - torch.eye(self.p)) - 2 * (Y[:, : idx].clone() - phi @ X_out[:, : idx]) @ torch.transpose(X_out[:, : idx], 0, 1))
        # To unit column vector
        phi_out = phi_out / torch.norm(phi_out, dim=0)
        #phi_out = phi.clone()

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
        self.X_old = X
        for idx in range(self.N):
            phi, X = self.inst_blocks[idx](phi, self.X_old, X, Y, idx)
        return phi, X
