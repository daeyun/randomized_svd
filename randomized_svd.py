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
    P = np.hstack((np.eye(k), proj))[:, np.argsort(idx)]
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
    A = S.copy()
    k = S.shape[1]
    assert k > 1 and start_col >= 0
    start_col = min(S.shape[1], start_col)
    if A.dtype != np.float32 and A.dtype != np.float64:
        A = A.astype(np.float64)

    if start_col == 0:
        A[:, 0] = normalize_vector(A[:, 0])

    uu = []
    for i in range(start_col + 1, k):
        A[:, i] = S[:, i]
        for j in range(0, i):
            u = A[:, j]
            v = A[:, i]
            if len(uu) <= j:
                uu.append(u.T.dot(u))
            A[:, i] -= u * (u.T.dot(v) / uu[j])

        A[:, i] = normalize_vector(A[:, i])
        # Re-project A[:, i] to the orthogonal complement of A[:, :i] to make sure they stay orthogonal.
        A[:, i] = A[:, i] - A[:, :i].dot(A[:, :i].T.dot(A[:, i]))

    return A


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


def check_orthogonal(Q):
    return la.norm(np.eye(Q.shape[1]) - Q.T.dot(Q), 2)


def adaptive_range(A, eps=1e-7, k=50, p=50, method='qr_merging', power_iter_k=2, power_iter_p=0, stabilize_qr_merging=True, max_p_iter=np.float('inf')):
    if A.shape[0] < 10 or A.shape[1] < 10:
        Q, _ = la.qr(A, mode='economic')
        return Q

    # Start with 50 columns by default.
    k = min(k, A.shape[1])

    omega = np.random.randn(A.shape[1], k)
    Q = A.dot(omega)  # N by k

    # Power iterations.
    for i in range(power_iter_k):
        Q, _ = la.lu(A.dot(A.T.dot(Q)), permute_l=True)
    Q, _ = la.qr(Q, mode='economic')

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

    Q, _ = la.qr(A.conj().T.dot(Q), mode='economic')
    Q, _ = la.qr(A.dot(Q), mode='economic')

    return Q


def randomized_svd(A):
    Q = adaptive_range(A)

    k = Q.shape[1]

    J, P = ID_row(Q, k)
    W, R = nla.qr(A[J, :].T)
    U, sigma, VT = nla.svd(P.dot(R.T), full_matrices=False)
    V = W.dot(VT.T).T

    return U, sigma, V


def pca(A, k=10):
    _, _, V = randomized_svd(A - A.mean(axis=0), None)
    return V[:, k]
