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
        ump = UMP(N, n, p, m, training=True, const=(0.1, 0.001, math.sqrt(2 * (-1 * -0.0001) / 0.01)))
        ump.fit(phi, X, Y)
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
