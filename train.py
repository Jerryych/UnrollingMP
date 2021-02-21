# Training script for algo.
import numpy as np
import h5py
from model import MP, UMP
import sys
import math
import numpy.linalg as LA
import torch


def load_data(dir, fname):
    path = dir + '//' + fname
    f = h5py.File(path, 'r')
    return f['X'][()], f['Y'][()], f['phi'][()], f['m'][()]

def main(dir, fname, model, mode):
    '''
    Load data and start training.
    '''
    X, Y, phi, m = load_data(dir, fname)
    p, N = X.shape
    n, _ = phi.shape
    if model == 'MP':
        mp = MP(N, n, p)
        mp.fit(Y, m, phi, X, mode=mode)
    elif model == 'UMP':
        # simple version training
        ump = UMP(N, n, p, m, training=False, const=(0.1, 0.001, math.sqrt(2 * (-1 * -0.0001) / 0.01)))
        #phi_init = np.random.uniform(-1.0, 1.0, size=(n, p))
        # To unit column vector
        #phi_init = phi_init / LA.norm(phi_init, axis=0)
        X_init = np.random.uniform(-1.0, 1.0, size=(p, N))
        ump.eval()
        phi_hat, X_hat = ump(torch.tensor(phi), torch.tensor(X_init), torch.tensor(Y))
        phi_hat = phi_hat.numpy()
        X_hat = X_hat.numpy()
        Y_diff = LA.norm(Y - phi_hat @ X_hat, ord='fro')
        # To 0/1 matrix
        match = np.where(X != 0, 1, 0) - np.where(X_hat != 0, 1, 0)
        x_sparse = sum(LA.norm(match, axis=0, ord=0))
        mu = LA.norm(phi_hat.T @ phi_hat - np.eye(p), ord='fro')
        print(f'Y diff: {Y_diff}, X s: {x_sparse}, mu: {mu}')
    else:
        print(f'No model named {model}')


if __name__ == '__main__':
    if len(sys.argv) == 5:
        dir = sys.argv[1]
        fname = sys.argv[2]
        model = sys.argv[3]
        m_mode = sys.argv[4]
        main(dir=dir, fname=fname, model=model, mode=m_mode)
    else:
        print('Not enough arguments...')
