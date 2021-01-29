import numpy as np


class MP:

    def __init__(self):

    def _init(self, N, n, p):
        # N: number of data instance
        # n: dimension of signal
        # p: number of atom
        self.phi = np.random.randn(n, p)
        self.x_p = np.random.randn(p, N)

    def fit(self, X, Y, m):
        # X: sparse representation
        # Y: signal
        # m: number of atom in sparse representation
        n, N = Y.shape
        p, _ = X.shape
        self._init(N, n, p)
        self.y = Y

        for i in range(m):
            for j in range(N):
                self.update_curr_x(i, j)
                self.update_past_x(i, j - 1)
                self.update_phi(i, j)

    def update_curr_x(self, m, j):

    def update_past_x(self, m, j):

    def udpate_phi(self, m, j):