# Generate orthogonal basis, coefficient (X), and signal (Y)
import numpy as np
import h5py
import os


def gen_basis(n, p):
    '''
    n: length of a basis
    p: number of basis
    '''
    basis = np.zeros((n, p))

    return basis

def gen_coeff(N, p, m):
    '''
    N: number of instance
    p: number of basis
    m: sparsity (non zeros)
    '''
    coeff = np.zeros((p, N))

    return coeff


def main(dir, N, n, p, m):
    '''
    N: number of instance
    n: length of a basis
    p: number of basis
    m: sparsity
    '''
    dir = dir + '//'

    phi = gen_basis(n, p)
    X = gen_coeff(N, p, m)
    Y = np.dot(phi, X)

    if not os.path.exists(dir):
        os.mkdir(dir)

    with h5py.File(dir + 'arti_dataset.hdf5', 'w') as f:
        f['phi'] = phi
        f['X'] = X
        f['Y'] = Y
        print('Write file: ' + f.name)


if __name__ == '__main__':
    main('dataset', 50, 16, 32, 4)