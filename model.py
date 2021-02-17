import numpy as np
from numpy import linalg as LA
import math


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
        self.phi = np.random.uniform(-1.0, 1.0, size=(n, p))
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

        if mode == 'm':
            self.fit_all_instances(m)
            p_d, x_d = self.eval(phi_real, X)
            print(f'Phi diff: {p_d}, X diff: {x_d}')
        elif mode == '1':
            for i in range(50):
                self.fit_all_instances(i)
                p_d, x_d = self.eval(phi_real, X)
                print(f'Phi diff: {p_d}, X diff: {x_d}')
                self.y = self.y - self.phi @ self.x_p

    def fit_all_instances(self, m):
        '''
        Fit (phi, X) given m
        '''
        for j in range(self.N):
            self.__update_curr_x(m, j)
            self.__update_past_x(m, j - 1)
            self.__update_phi(j)

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
        self.phi = self.phi - 0.0001 * self.__grad_phi(j)

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
        measure difference between estimated (phi, X) and real (phi, X) with 2-norm
        '''
        phi_diff = LA.norm(phi_real - self.phi)
        X_diff = LA.norm(X - self.x_p)
        if Y:
            Y_diff = LA.norm(Y - self.phi @ self.x_p)
            return phi_diff, X_diff, Y_diff
        else:
            return phi_diff, X_diff


class UMP:

    def __init__(self, N, n, p, m):
        '''
        N: number of data instance
        n: dimension of signal
        p: number of atom
        m: number of atom in sparse representation
        '''

    def _instance_block(self, m, j):
        pass

    def _constraint_block(self, m):
        pass

    def forward(self, N, n, p, m):
        pass
