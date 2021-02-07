import numpy as np
from numpy import linalg as LA


class MP:

    def __init__(self, N, n, p):
        '''
        N: number of data instance
        n: dimension of signal
        p: number of atom
        '''
        self.phi = np.random.randn(n, p)
        self.x_p = np.random.randn(p, N)
        self.N = N
        self.n = n
        self.p = p

    def fit(self, Y, m):
        '''
        fit (phi, X)

        X: sparse representation
        Y: signal
        m: number of atom in sparse representation
        '''
        self.y = Y

        for i in range(m):
            for j in range(self.N):
                self.update_curr_x(i, j)
                self.update_past_x(i, j - 1)
                self.update_phi(i, j)

    def update_curr_x(self, m, j):

    def update_past_x(self, m, j):

    def udpate_phi(self, m, j):

    def eval(self, phi_real, X):
        '''
        measure difference between estimated (phi, X) and real (phi, X) with 2-norm
        '''
        phi_diff = LA.norm(phi_real - self.phi)
        X_diff = LA.norm(X - self.x_p)
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

    def _constraint_block(self, m):

    def forward(self, N, n, p, m):
