# Generate orthogonal basis, coefficient (X), and signal (Y)
import numpy as np
from numpy import linalg as LA
import h5py
import os
import sys


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
        # normalized cosine basis
        vec = np.cos(t_k)
        norm = LA.norm(vec)
        if norm != 0:
            vec = vec / norm
        C[:, k] = vec
        # normalized sine basis
        vec = np.sin(t_k)
        norm = LA.norm(vec)
        if norm != 0:
            vec = vec / norm
        S[:, k] = vec
    basis = np.concatenate((C, S), axis=1)
    return basis

def gen_basis_compact(n, p):
    if n % 2 == 0:
        bn = n / 2 + 1
    else:
        bn = (n + 1) / 2
    bn = int(bn)
    C = np.zeros((n, bn))
    S = np.zeros((n, bn))
    ns = np.arange(1, n + 1)
    one_cycle = 2 * np.pi * ns / n
    for k in range(bn):
        t_k = k * one_cycle
        # normalized cosine basis
        vec = np.cos(t_k)
        norm = LA.norm(vec)
        if norm != 0:
            vec = vec / norm
        C[:, k] = vec
        # normalized sine basis
        vec = np.sin(t_k)
        norm = LA.norm(vec)
        if norm != 0:
            vec = vec / norm
        S[:, k] = vec
    basis = np.concatenate((C, S[:, : -2]), axis=1)
    return basis

def gen_coeff(N, p, m):
    '''
    N: number of instance
    p: number of basis
    m: sparsity (non zeros)
    '''
    coeff = np.zeros((p, N))
    for i in range(N):
        pos_w = list(zip(np.random.choice(p, m), np.random.uniform(1, 2, m)))
        column = np.zeros(p)
        for pos, w in pos_w:
            column[pos] = w
        coeff[:, i] = column
    return coeff

def main(dir, fname, N, n, p, m, compact=False):
    '''
    N: number of instance
    n: length of a basis
    p: number of basis
    m: sparsity
    '''
    dir = dir + '//'

    if not compact:
        phi = gen_basis(n, p)
        X = gen_coeff(N, p, m)
    else:
        phi = gen_basis_compact(n, p)
        X = gen_coeff(N, phi.shape[1], m)
    Y = np.matmul(phi, X)

    if not os.path.exists(dir):
        os.mkdir(dir)

    with h5py.File(dir + fname, 'w') as f:
        f['phi'] = phi
        f['X'] = X
        f['Y'] = Y
        f['m'] = m
        print('Write file: ' + f.filename)


if __name__ == '__main__':
    if len(sys.argv) == 8:
        dir = sys.argv[1]
        fname = sys.argv[2]
        N = int(sys.argv[3])
        n = int(sys.argv[4])
        p =  int(sys.argv[5])
        m = int(sys.argv[6])
        c = bool(sys.argv[7])
        main(dir, fname, N, n, p, m, compact=c)
        # main('dataset', 'arti_dataset_compact.hdf5', 50, 16, 32, 4, compact=True)
    else:
        print('Not enough arguments...')
