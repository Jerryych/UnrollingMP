# Training script for algo.
import numpy as np
import h5py
from model import MP


def load_data(dir, fname):
    path = dir + '//' + fname
    f = h5py.File(path, 'r')
    return f['X'][()], f['Y'][()], f['phi'][()], f['m'][()]

def main(dir, fname, model):
    '''
    Load data and start training.
    '''
    X, Y, phi, m = load_data(dir, fname)
    p, N = X.shape
    n, _ = phi.shape
    if model == 'MP':
        mp = MP(N, n, p)
        mp.fit(Y, m, phi, X)
    elif model == 'UMP':
        pass
    else:
        print(f'No model named {model}')


if __name__ = '__main__':
    main(dir='dataset', fname='arti_dataset.hdf5', model='MP')
