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

    def fit(self, Y, m, phi_real, X):
        '''
        fit (phi, X)

        X: sparse representation
        Y: signal
        m: number of atom in sparse representation
        '''
        self.y = Y

        for i in range(m):
            for j in range(self.N):
                self.__update_curr_x(i, j)
                self.__update_past_x(i, j - 1)
                self.__update_phi(i, j)
            p_d, x_d = self.eval(phi_real, X)
            print(f'Phi diff: {p_d}, X diff: {x_d}')

    def __update_curr_x(self, m, j):

    def __update_past_x(self, m, j):

    def __udpate_phi(self, m, j):

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
