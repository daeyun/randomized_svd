from __future__ import division

import numpy as np
import scipy.linalg as la
import scipy.linalg.interpolative as sli
import matplotlib.pyplot as pt
import scipy
import time
import numpy.linalg as nla
import warnings
import sys
import collections


def timeit(func):
    func()
    start = time.time()
    for i in range(10):
        func()
    end = time.time()
    return (end - start) / 10.0


def ID(A, k):
    """
    A_{m,n}=A[:,J] P

    :return P: k by n
    """
    idx, proj = sli.interp_decomp(A, k)
    J = idx[:k]

    kp = np.eye(k)
    if len(proj):
        kp = np.hstack((kp, proj))

    P = kp[:, np.argsort(idx)]
    return J, P


def ID_row(A, k):
    """
    A_{m,n}=P A[J,:]

    :return P: n by k
    """
    J, P = ID(A.T, k)
    return J, P.T


def proj(u, v):
    return u * (u.T.dot(v) / u.T.dot(u))


def normalize_vector(v):
    return v / la.norm(v, 2)


def gram_schmidt(S, start_col=0):
    """
    Implementation of gram schmidt that assumes first start_col+1 columns are already orthogonal.
    :return:
    """
    Q = S.copy()
    k = S.shape[1]
    assert k > 1 and start_col >= 0
    start_col = min(S.shape[1], start_col)
    if Q.dtype != np.float32 and Q.dtype != np.float64:
        Q = Q.astype(np.float64)

    if start_col == 0:
        Q[:, 0] = normalize_vector(Q[:, 0])

    uu = []
    for i in range(start_col + 1, k):
        Q[:, i] = S[:, i]
        for j in range(0, i):
            u = Q[:, j]
            v = Q[:, i]
            if len(uu) <= j:
                uu.append(u.T.dot(u))
            Q[:, i] -= u * (u.T.dot(v) / uu[j])

        Q[:, i] = normalize_vector(Q[:, i])
        # Re-project Q[:, i] to the orthogonal complement of Q[:, :i] to make sure they stay orthogonal.
        Q[:, i] = Q[:, i] - Q[:, :i].dot(Q[:, :i].T.dot(Q[:, i]))

    return Q


def lowrank(self, n=100, m=100):
    A = np.random.randn(n, n)
    U, s, Vh = la.svd(A)
    sigma = np.exp(-np.arange(n))
    return (U * sigma).dot(Vh)


def svd_err(svd_func, A=lowrank(100, 100), **kwargs):
    U, s, Vh = svd_func(A, **kwargs)
    err = la.norm((U * s).dot(Vh) - A, 2)
    return err


def fast_2_norm(A):
    """
    Uses |A*v|_2 to estimate |A|_2
    A*v shifts v towards the largest singular vector.
    """
    v = np.random.rand(A.shape[1], 1)
    return la.norm(A.dot(v))


def normalize_columns(A):
    return A / np.sqrt(np.power(A, 2).sum(axis=0))


def randomized_range(A, rank, power_iter=2):
    """
    Uses fixed rank i.e. non-adaptive version.

    :param rank: Approximate rank.
    :param power_iter: Number of power iterations.
    :return: Orthogonal basis Q. Low rank approximation of A.
    """
    Q = np.random.randn(A.shape[1], rank)

    # Power iterations.
    for i in range(power_iter):
        Q, _ = la.lu(A.T.dot(A.dot(Q)), permute_l=True)
    Q, _ = la.qr(A.dot(Q), mode='economic')

    return Q


def adaptive_range(A, eps=1e-7, k=50, p=50, method='qr_merging', power_iter_k=2, power_iter_p=0,
                   stabilize_qr_merging=True, max_p_iter=np.float('inf')):
    if A.shape[0] < 10 or A.shape[1] < 10:
        Q, _ = la.qr(A, mode='economic')
        return Q

    # Start with 50 columns by default.
    k = min(k, A.shape[1])

    Q = np.random.randn(A.shape[1], k)

    # Power iterations.
    for i in range(power_iter_k):
        Q, _ = la.lu(A.T.dot(A.dot(Q)), permute_l=True)

    Q, _ = la.qr(A.dot(Q), mode='economic')

    num_prev_errs = 8
    errs = collections.deque(maxlen=num_prev_errs)
    sizes = collections.deque(maxlen=num_prev_errs)

    IQQA = (A - Q.dot(Q.T.dot(A)))
    round = 0
    while True:
        round += 1
        err = fast_2_norm(IQQA)

        errs.append(abs(err))
        sizes.append(Q.shape[1])

        if (eps is not None and err <= eps) or Q.shape[1] >= A.shape[1] or round > max_p_iter:
            break
        elif eps is None and len(errs) > 3 and np.std(errs) > 1e4 * np.std(list(errs)[-3:]) and err < 0.1:
            # Terminate if convergence is detected.
            ind = min((err, idx) for idx, err in enumerate(errs))[1]
            Q = Q[:, :sizes[ind]]
            break

        ncols = p if Q.shape[1] + p <= A.shape[1] else A.shape[1] - Q.shape[1]

        Q_ = A.dot(np.random.randn(A.shape[1], ncols))

        # For experiments. This doesn't seem to help much.
        for i in range(power_iter_p):
            Q_, _ = la.lu(A.dot(A.T.dot(Q_)), permute_l=True)

        if method == 'qr_merging':
            if stabilize_qr_merging:
                Q_, _ = la.qr((Q_ - Q.dot(Q.T.dot(Q_))), mode='economic')
            # Re-project to orthogonal complement for stability.
            Q_, R = la.qr((Q_ - Q.dot(Q.T.dot(Q_))), mode='economic')

            # If the new columns are not full rank.
            if np.abs(np.diag(R)).min() < 1e-4:
                # Exclude ones not orthogonal to existing columns.
                inds = (np.abs(Q.T.dot(Q_)).max(axis=0) < 1e-4)
                Q_ = Q_[:, inds]
                if sum(inds) == 0:
                    break

            Q = np.concatenate((Q, Q_), axis=1)
            IQQA -= Q_.dot(Q_.T.dot(A))

        elif method == 'modified_gram_schmidt':
            last_col = Q.shape[1] - 1
            Q_ = normalize_columns(Q_)
            Q = np.concatenate((Q, Q_), axis=1)
            Q = gram_schmidt(Q, start_col=last_col)
            IQQA -= Q[:, last_col + 1:].dot(Q[:, last_col + 1:].T.dot(A))
        else:
            raise RuntimeError('unknown method')

    return Q


def randomized_svd(A, rank=None, power_iter=2, k=50, p=50):
    if rank is None:
        Q = adaptive_range(A, k=k, p=p, power_iter_k=power_iter)
    else:
        Q = randomized_range(A, rank, power_iter=power_iter)

    k = Q.shape[1]

    J, P = ID_row(Q, k)
    W, R = la.qr(A[J, :].T, mode='economic')
    U, sigma, VT = la.svd(P.dot(R.T), full_matrices=False)
    V = VT.dot(W.T)

    return U, sigma, V


def pca(A, num_components=10, svd_rank=None):
    _, _, V = randomized_svd(A - A.mean(axis=0), svd_rank)
    return V[:, num_components]
