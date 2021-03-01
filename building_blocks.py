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

    def forward(self, phi, X_old, X_prv, Y, idx):
        X_out = torch.zeros_like(X_old)
        # Update X[idx] with X_old[idx] from previous Constraint_block in vector form
        x_vec = X_old[:, idx].clone() - (-2) * self.step * torch.transpose(phi, 0, 1) @ (Y[:, idx].clone() - phi @ X_old[:, idx].clone())
        X_out[:, idx] = self.thresh_func(x_vec)
        # Update X[0] ~ X[idx - 1] with X_prv[0] ~ X_prv[idx - 1] from previous Instance_block in matrix form
        if idx != 0:
            x_tmp = X_prv[:, : idx].clone() - (-2) * self.step * torch.transpose(phi, 0, 1) @ (Y[:, : idx].clone() - phi @ X_prv[:, : idx].clone())
            X_out[:, : idx] = self.thresh_func(x_tmp)
        # Update phi with X[0] ~ X[idx]
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
