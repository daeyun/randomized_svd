import numpy as np
import scipy.linalg as la
import scipy.linalg.interpolative as sli
import matplotlib.pyplot as pt
import scipy
import time
import numpy.linalg as nla


def scipy_rand_svd():
    A = np.random.randn(500, 500)
    return sli.svd(A, 30)


def scipy_svd():
    A = np.random.randn(500, 500)
    return la.svd(A)


def timeit(func):
    func()
    start = time.time()
    for i in range(10):
        func()
    end = time.time()
    return (end - start) / 10.0


def ID(A, k):
    '''
    A_{m,n}=A[:,J] P

    :return P: k by n
    '''
    idx, proj = sli.interp_decomp(A, k)
    J = idx[:k]
    P = np.hstack((np.eye(k), proj))[:, np.argsort(idx)]
    return J, P


def ID_row(A, k):
    '''
    A_{m,n}=P A[J,:]

    :return P: n by k
    '''
    J, P = ID(A.T, k)
    return J, P.T


def proj(u, v):
    return u * np.inner(v, u) / np.inner(u, u)


def gs(S):
    # S = {v1, v2, v3,..., vk}
    S_ = S.copy()
    k = S.shape[1]
    for i in range(k):
        vi = S_[:, i] / np.sqrt(np.power(S_[:, i], 2).sum())
        for j in range(i + 1, k):
            S_[:, j] -= proj(vi, S_[:, j])
    return S_


def lowrank(self, n=100, m=100, k=10):
    A = np.random.randn(n, n)
    U, s, Vh = la.svd(A)
    sigma = np.exp(-np.arange(n))
    return (U * sigma).dot(Vh)


def svd_err(svd_func, A=lowrank(100, 100), **kwargs):
    U, s, Vh = svd_func(A, **kwargs)
    err = la.norm((U * s).dot(Vh) - A, 2)
    return err


def fast_2_norm(A):
    '''
    Uses |A*v|_2 to estimate |A|_2
    A*v shifts v towards the largest singular vector.
    From homework 2
    '''
    v = np.random.rand(A.shape[0], 1)
    return la.norm(A.dot(v))


def adaptive_range(A, eps=1e-9):
    # Also mostly from homework
    N = A.shape[0]

    # Start with 10 columns.
    k = min(10, A.shape[1])

    # Add 20 more each time.
    p = 20

    omega = np.random.randn(A.shape[0], k)
    Y = A.dot(omega)  # N by k

    iter = 0
    while True:
        # Probably possible to just do a few additional rounds of Gram–Schmidt here
        # instead of computing Q from scratch.
        if iter==0:
            Q, _ = la.qr(Y, mode='economic')  # O(N(k+p)^2)
        else:
            Q, _ = la.lu(Y, permute_l=True)
        iter += 1

        E = (np.eye(Q.shape[0]) - Q.dot(Q.T)).dot(A)
        err = fast_2_norm(E)  # O(N)
        if err <= eps:
            return Q
        Y = np.concatenate((Y, A.dot(np.random.randn(N, p))), axis=1)


def compute_basis(A, k=None, iter=2):
    if k is None:
        adaptive_range()
    else:
        omega = np.random.randn(A.shape[1], k)
        Y = A.dot(omega)
        Q, _ = nla.qr(Y)

    for _ in range(iter):
        Q, _ = la.lu(A.conj().T.dot(Q), permute_l=True)
        Q, _ = la.lu(A.dot(Q), permute_l=True)

    return Q


def randomized_svd(A, k=10, compute_Q=True, iter=4):
    if compute_Q:
        Q = compute_basis(A, k, iter=iter)
    else:
        omega = np.random.randn(A.shape[1], k)
        Q = A.dot(omega)

    J, P = ID_row(Q, k)
    W, R = nla.qr(A[J, :].T)
    U, sigma, VT = nla.svd(P.dot(R.T), full_matrices=False)
    V = W.dot(VT.T).T
    # U, s, Vh = la.svd(A)
    return U, sigma, V


def pca(A, k=10):
    _, _, V = randomized_svd(A - A.mean(axis=0), k)
    return V

# Q = Q[:, :10]
#
# k = Q.shape[1]
#
# J, P = ID_row(Q, k)
#
# Q_, R_ = la.qr(A[J, :].T)
#
# U, sigma, VT_ = la.svd(P.dot(R_.T), full_matrices=False)
#
# VT = Q_.dot(VT_.T).T
#
# print(U.shape)
# print(sigma.shape)
# print(VT.shape)
