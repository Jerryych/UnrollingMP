# Generate orthogonal basis, coefficient (X), and signal (Y)
import numpy as np
import h5py
import os


def gen_basis(n, p):
    '''
    n: length of a basis
    p: number of basis
    '''
    C = np.zeros((n, n))
    S = np.zeros((n, n))
    ns = np.arange(n)
    one_cycle = 2 * np.pi * ns / n
    for k in range(n):
        t_k = k * one_cycle
        C[:, k] = np.cos(t_k)
        S[:, k] = np.sin(t_k)
    basis = np.concatenate((C, S), axis=1)
    return basis

def gen_coeff(N, p, m):
    '''
    N: number of instance
    p: number of basis
    m: sparsity (non zeros)
    '''
    coeff = np.zeros((p, N))
    for i in range(N):
        pos_w = list(zip(np.random.choice(p, m), np.random.rand(m)))
        column = np.zeros(p)
        for pos, w in pos_w:
            column[pos] = w
        coeff[:, i] = column
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
    Y = np.matmul(phi, X)

    if not os.path.exists(dir):
        os.mkdir(dir)

    with h5py.File(dir + 'arti_dataset.hdf5', 'w') as f:
        f['phi'] = phi
        f['X'] = X
        f['Y'] = Y
        f['m'] = m
        print('Write file: ' + f.name)


if __name__ == '__main__':
    main('dataset', 50, 16, 32, 4)
