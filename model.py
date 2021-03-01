import numpy as np
from numpy import linalg as LA
import math
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from building_blocks import Instance_block, Constraint_block


class MP:

    def __init__(self, N, n, p):
        '''
        N: number of data instance
        n: dimension of signal
        p: number of atom
        step: step size for update
        rho: lagrange multiplier for constraint m
        lam: hyper-param for proximal gradient
        threshold: shrinkage operator
        '''
        #self.phi = np.random.uniform(-1.0, 1.0, size=(n, p))
        # To unit column vector
        #self.phi = self.phi / LA.norm(self.phi, axis=0)
        self.x_p = np.random.uniform(-1.0, 1.0, size=(p, N))
        self.N = N
        self.n = n
        self.p = p
        self.step = 0.1
        self.rho = -0.0001
        self.lam = 0.01
        self.threshold = math.sqrt(2 * (-1 * self.rho) / self.lam)

    def fit(self, Y, m, phi_real, X, mode='m'):
        '''
        Fit (phi, X)

        X: sparse representation
        Y: signal
        m: number of atom in sparse representation
        mode: 'm' for 1 fit_all_instances() with constraint m, '1' for m fit_all_instances() with increasing constraint
        '''
        self.y = Y
        self.phi = phi_real

        if mode == 'm':
            self.fit_all_instances(m)
            p_d, x_d, y_d = self.eval(phi_real, X, Y=Y)
            print(f'Phi diff: {p_d}, X diff: {x_d}, Y diff: {y_d}')
            y_d, x_mis, mu = self.objective_func(phi_real, X, Y, m)
            print(f'OBJ Y diff: {y_d}, X mismatch: {x_mis}, mu: {mu}')
        elif mode == '1':
            #p_d, x_d, y_d = self.eval(phi_real, X, Y=Y)
            #print(f'Init    Phi diff: {p_d}, X diff: {x_d}, Y diff: {y_d}')
            y_d, x_mis, mu = self.objective_func(phi_real, X, Y, m)
            print(f'Init    Y diff: {y_d}, X mismatch: {x_mis}, mu: {mu}')
            for i in range(m):
                self.fit_all_instances(i)
                #p_d, x_d, y_d = self.eval(phi_real, X, Y=Y)
                #print(f'{i} Phi diff: {p_d}, X diff: {x_d}, Y diff: {y_d}')
                y_d, x_mis, mu = self.objective_func(phi_real, X, Y, m)
                print(f'{i} Y diff: {y_d}, X mismatch: {x_mis}, mu: {mu}')
                #self.y = self.y - self.phi @ self.x_p

    def fit_all_instances(self, m):
        '''
        Fit (phi, X) given m
        '''
        for j in range(self.N):
            self.__update_curr_x(m, j)
            self.__update_past_x(m, j - 1)
            #self.__update_phi(j)

    def __update_curr_x(self, m, j):
        '''
        Proximal gradient update of j-th column of x
        '''
        self.x_p[:, j] = self.x_p[:, j] - self.__grad_x(j)
        self.x_p[:, j] = self.__prox_grad_x(m, j)

    def __update_past_x(self, m, j):
        '''
        Proximal gradient update of x from 0 to j - 1 (included)
        Column by column update
        '''
        j = j + 1
        for idx in range(j):
            self.x_p[:, idx] = self.x_p[:, idx] - self.__grad_x(idx)
            self.x_p[:, idx] = self.__prox_grad_x(m, idx)

    def __update_phi(self, j):
        '''
        Update phi with {x[0] ~ x[j]}
        Matrix update
        '''
        self.phi = self.phi - 0.001 * self.__grad_phi(j)
        # To unit column vector
        self.phi = self.phi / LA.norm(self.phi, axis=0)

    def __grad_x(self, j):
        '''
        In vector form
        d_x = phi' * (y - phi * x)
        '''
        return -1 * self.step * self.phi.T @ (self.y[:, j] - self.phi @ self.x_p[:, j])

    def __prox_grad_x(self, m, j):
        '''
        In vector form
        Hard thresholding
        '''
        x = self.x_p[:, j]
        return np.where(np.absolute(x) > self.threshold, x, 0)

    def __grad_phi(self, j):
        '''
        In matrix form
        d_phi = 4 * phi * (phi' * phi - I) - 2 * (y - phi * x) * x'
        '''
        j = j + 1
        return 4 * self.phi @ (self.phi.T @ self.phi - np.eye(self.p)) - 2 * (self.y[:, : j] - self.phi @ self.x_p[:, : j]) @ self.x_p[:, : j].T

    def get_X(self):
        return self.x_p

    def get_phi(self):
        return self.phi

    def eval(self, phi_real, X, Y=None):
        '''
        Measure difference between estimated (phi, X) and real (phi, X) with 2-norm
        If Y is an object, return difference of estimated Y and real Y
        '''
        phi_diff = LA.norm(phi_real - self.phi)
        X_diff = LA.norm(X - self.x_p)
        if Y is not None:
            Y_diff = LA.norm(Y - self.phi @ self.x_p)
            return phi_diff, X_diff, Y_diff
        else:
            return phi_diff, X_diff

    def objective_func(self, phi_real, X, Y, m):
        '''
        Measure objective function
        |y - PHI * x|_2 ^ 2 + rho * (m - |x|_0) + |PHI' * PHI - I|_2 ^ 2
        '''
        Y_diff = LA.norm(Y - self.phi @ self.x_p, ord='fro')
        # To 0/1 matrix
        mismatch = np.where(X != 0, 1, 0) - np.where(self.x_p != 0, 1, 0)
        x_mis = sum(LA.norm(mismatch, axis=0, ord=0))
        mu = LA.norm(self.phi.T @ self.phi - np.eye(self.p), ord='fro')
        return Y_diff, x_mis, mu


class UMP(nn.Module):

    def __init__(self, N, n, p, m, training=False, const=None):
        '''
        N: number of data instance
        n: dimension of signal
        p: number of atom
        m: number of atom in sparse representation
        training: with parameter training if it's set to True
        const: tuple of parameter init values (steps for X update, phi_steps, thresholds for prox grad)
        '''
        super(UMP, self).__init__()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('UMP using device:', self.dev)
        if self.dev.type == 'cuda':
            self.to(self.dev)
        self.N = N
        self.n = n
        self.p = p
        self.m = m
        self.training = training
        trs, steps, phi_steps, threshs = self.__create_var(self.N, self.m, training, const)
        layers = [Constraint_block(self.N, self.p, trs[i], steps[i, :], phi_steps[i, :], threshs[i, :]) for i in range(self.m)]
        self.const_blocks = nn.ModuleList(layers)

    def __create_var(self, N, m, training, const):
        trs = np.full(m, training).tolist()
        if const is not None:
            steps = np.full((m, N), const[0])
            phi_steps = np.full((m, N), const[1])
            threshs = np.full((m, N), const[2])
        else:
            steps = np.random.uniform(0.0, 0.1, size=(m, N))
            phi_steps = np.random.uniform(0.0, 0.1, size=(m, N))
            threshs = np.random.uniform(0.0, 0.1, size=(m, N))
        return trs, torch.tensor(steps), torch.tensor(phi_steps), torch.tensor(threshs)

    def forward(self, phi, X, Y):
        for idx in range(self.m):
            phi, X = self.const_blocks[idx](phi, X, Y)
        return phi, X

    def fit(self, phi, X, Y):
        #phi_init = np.random.uniform(-1.0, 1.0, size=(self.n, self.p))
        # To unit column vector
        #phi_init = phi_init / LA.norm(phi_init, axis=0)
        phi_init = phi
        X_init = np.random.uniform(-1.0, 1.0, size=(self.p, self.N))
        if self.training:
            opt = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
            loss_func = F.mse_loss
            self.__fit(*(phi, phi_init), *(X, X_init), Y, 50, loss_func, opt)
        else:
            self.eval()
            phi_hat, X_hat = self(torch.as_tensor(phi_init), torch.as_tensor(X_init), torch.as_tensor(Y))
            phi_hat = phi_hat.numpy()
            X_hat = X_hat.numpy()
            Y_diff, x_mis, mu = self.objective_func(*(phi, phi_hat), *(X, X_hat), Y)
            print(f'Y diff: {Y_diff}, X mismatch: {x_mis}, mu: {mu}')

    def __fit(self, phi, phi_init, X, X_init, Y, epochs, loss_func, opt):
        if self.dev.type == 'cuda':
            phi = torch.tensor(phi).cuda()
            phi_init = torch.tensor(phi_init).cuda()
            X = torch.tensor(X).cuda()
            X_init = torch.tensor(X_init).cuda()
            Y = torch.tensor(Y).cuda()
        else:
            phi = torch.tensor(phi)
            phi_init = torch.tensor(phi_init)
            X = torch.tensor(X)
            X_init = torch.tensor(X_init)
            Y = torch.tensor(Y)
        for epoch in range(epochs):
            self.train()
            phi_hat, X_hat = self(phi_init, X_init, Y)
            loss = loss_func(X_hat, X)

            loss.backward()
            opt.step()
            opt.zero_grad()

            self.eval()
            with torch.no_grad():
                phi_hat, X_hat = self(phi_init, X_init, Y)
                v_loss = loss_func(X_hat, X)
                if self.dev.type == 'cuda':
                    Y_diff, x_mis, mu = self.objective_func(*(phi.cpu().numpy(), phi_hat.cpu().numpy()), *(X.cpu().numpy(), X_hat.cpu().numpy()), Y.cpu().numpy())
                else:
                    Y_diff, x_mis, mu = self.objective_func(*(phi.numpy(), phi_hat.numpy()), *(X.numpy(), X_hat.numpy()), Y.numpy())
                print(f'{epoch:2}| Y diff: {Y_diff:8.4f}| X mismatch: {x_mis:8.4f}| X mse: {v_loss.numpy():8.4f}| mu: {mu:8.4f}')

    def objective_func(self, phi_real, phi_hat, X, X_hat, Y):
        '''
        Measure objective function
        |y - PHI * x|_2 ^ 2 + rho * (m - |x|_0) + |PHI' * PHI - I|_2 ^ 2
        '''
        Y_diff = LA.norm(Y - phi_hat @ X_hat, ord='fro')
        # To 0/1 matrix
        mismatch = np.where(X != 0, 1, 0) - np.where(X_hat != 0, 1, 0)
        x_mis = sum(LA.norm(mismatch, axis=0, ord=0))
        mu = LA.norm(phi_hat.T @ phi_hat - np.eye(self.p), ord='fro')
        return Y_diff, x_mis, mu
