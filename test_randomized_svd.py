import unittest
import numpy as np
import scipy.linalg as la
import randomized_svd
import timeit


def lowrank(self, n=100, m=100, k=10):
    A = np.random.randn(n, n)
    U, s, Vh = la.svd(A)
    sigma = np.exp(-np.arange(n))
    return (U * sigma).dot(Vh)


class TestRandomizedSvd(unittest.TestCase):
    def full_svd(self, A):
        U, s, Vh = la.svd(A)
        return U, s, Vh

    def test_accurate_enough(self):
        A = lowrank(100, 80)
        U, s, Vh = randomized_svd.randomized_svd(A, 30)
        self.assertLess(la.norm((U * s).dot(Vh) - A, 2), 1e-10)

    def test_less_accurate_than_full_svd(self):
        A = lowrank(100, 80)

        U, s, Vh = randomized_svd.randomized_svd(A, 20)
        S = la.diagsvd(s, U.shape[1], U.shape[1])
        randomized_err = la.norm(U.dot(S).dot(Vh) - A, 2)

        U, s, Vh = self.full_svd(A)
        S = la.diagsvd(s, U.shape[1], U.shape[1])
        full_err = la.norm(U.dot(S).dot(Vh) - A, 2)

        self.assertGreater(1e-2 * randomized_err, full_err)

    def test_faster(self):
        n = 10
        full = timeit.timeit("svd(randn(500, 500))",
                             ("from scipy.linalg import svd;"
                              "from numpy.random import randn"), number=n)
        fast = timeit.timeit("randomized_svd(randn(500, 500))",
                             ("from randomized_svd import randomized_svd;"
                              "from numpy.random import randn"), number=n)
        self.assertLess(fast, full / 2.0)
