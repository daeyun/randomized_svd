import unittest
import numpy as np
import scipy.linalg as la
import randomized_svd
import timeit
import sys


def lowrank(self, n=100, m=100, k=10):
    A = np.random.randn(n, n)
    U, s, Vh = la.svd(A)
    sigma = np.exp(-np.arange(n))
    return (U * sigma).dot(Vh)


class TestRandomizedSvd(unittest.TestCase):
    def full_svd(self, A):
        U, s, Vh = la.svd(A)
        return U, s, Vh

    def test_adaptive_range_small(self):
        A = np.random.randn(5, 8)
        Q = randomized_svd.adaptive_range(A, 1e-9)
        self.assertEqual(Q.shape, (5, 5))

    def test_adaptive_range_big(self):
        A = np.random.randn(500, 400)
        Q = randomized_svd.adaptive_range(A, 1e-9)
        self.assertEqual(Q.shape, (500, 400))

    def test_adaptive_range_low_rank(self):
        A = np.random.randn(500, 100).dot(np.random.randn(100, 500))
        Q = randomized_svd.adaptive_range(A, 1e-9)
        self.assertEqual(Q.shape, (500, 100))

    def test_adaptive_range_low_rank_2(self):
        A = np.random.randn(50, 30).dot(np.random.randn(30, 50))
        Q = randomized_svd.adaptive_range(A, 1e-9)
        self.assertGreater(Q.shape[1], 20)

    def test_adaptive_range_low_rank_3(self):
        A = np.random.randn(1400, 500).dot(np.random.randn(500, 1400))
        Q = randomized_svd.adaptive_range(A, 1e-6)
        self.assertEqual(Q.shape[1], 500)

    def test_adaptive_range_low_rank_4(self):
        A = np.random.randn(1400, 550).dot(np.random.randn(550, 1400))
        Q = randomized_svd.adaptive_range(A, 1e-6)
        self.assertEqual(Q.shape[1], 550)

    def test_adaptive_range_low_rank_5(self):
        A = np.random.randn(200, 150).dot(np.random.randn(150, 200))
        Q = randomized_svd.adaptive_range(A, 1e-6, method='modified_gram_schmidt')
        self.assertEqual(Q.shape[1], 150)

    def test_adaptive_range_nonsquare_horz(self):
        A = np.random.randn(10, 50)
        Q = randomized_svd.adaptive_range(A, 1e-9)
        self.assertEqual(Q.shape, (10, 10))

    def test_gram_schmidt_shape(self):
        A = np.random.randn(20, 10)
        V = randomized_svd.gram_schmidt(A)
        self.assertEqual((20, 10), V.shape)

        A = np.random.randn(20, 30)
        V = randomized_svd.gram_schmidt(A)
        self.assertEqual((20, 30), V.shape)

    def test_gram_schmidt_orthonormal(self):
        A = np.random.randn(20, 10)
        V = randomized_svd.gram_schmidt(A)
        self.assertLess((np.eye(10) - V.T.dot(V)).max(), 1e-12)

    def test_gram_schmidt_orthonormal_2(self):
        A = np.random.randn(100, 100)
        V = randomized_svd.gram_schmidt(A)
        self.assertLess((np.eye(100) - V.T.dot(V)).max(), 1e-12)

    def test_gram_schmidt_use_start_column(self):
        A = np.random.randn(50, 5)
        Q, _ = la.qr(A, mode='economic')
        A = np.hstack((Q, np.random.randn(50, 5)))
        V = randomized_svd.gram_schmidt(A, start_col=4)
        self.assertLess((np.eye(5) - V[:, 5:].T.dot(V[:, 5:])).max(), 1e-10)

    def test_gram_schmidt_use_start_column2(self):
        A = np.random.randn(50, 10)
        Q, R = la.qr(A, mode='economic')
        Q = np.hstack((Q, np.random.randn(50, 30)))
        V = randomized_svd.gram_schmidt(Q, start_col=9)
        self.assertLess((np.eye(40) - V.T.dot(V)).max(), 1e-10)

    def test_range_using_modified_gram_schmidt(self):
        A = np.random.randn(50, 10).dot(np.random.randn(10, 50))
        Q = randomized_svd.adaptive_range(A, power_iter_k=0, power_iter_p=0, method='modified_gram_schmidt', k=5, p=10,
                                          eps=1e-7)
        self.assertLess(la.norm(A - Q.dot(Q.T.dot(A)), 2), 1e-7)

    @unittest.skip('not implemented')
    def test_gram_schmidt_in_place(self):
        A = np.random.randn(50, 10)
        randomized_svd.gram_schmidt(A, in_place=True)
        self.assertLess((np.eye(10) - A.T.dot(A)).max(), 1e-10)

    @unittest.skip('not implemented')
    def test_gram_schmidt_not_in_place(self):
        A = np.random.randn(50, 10)
        B = A.copy()
        V = randomized_svd.gram_schmidt(A)
        self.assertLess((np.eye(10) - V.T.dot(V)).max(), 1e-10)
        self.assertLess((A - B).max(), 1e-10)

    @unittest.skip('not implemented')
    def test_accurate_enough(self):
        A = lowrank(100, 80)
        U, s, Vh = randomized_svd.randomized_svd(A, 30)
        self.assertLess(la.norm((U * s).dot(Vh) - A, 2), 1e-10)

    @unittest.skip('not implemented')
    def test_less_accurate_than_full_svd(self):
        A = lowrank(100, 80)

        U, s, Vh = randomized_svd.randomized_svd(A, 20)
        S = la.diagsvd(s, U.shape[1], U.shape[1])
        randomized_err = la.norm(U.dot(S).dot(Vh) - A, 2)

        U, s, Vh = self.full_svd(A)
        S = la.diagsvd(s, U.shape[1], U.shape[1])
        full_err = la.norm(U.dot(S).dot(Vh) - A, 2)

        self.assertGreater(1e-2 * randomized_err, full_err)

    @unittest.skip('not implemented')
    def test_runtime(self):
        n = 5
        full = timeit.timeit("svd(randn(500, 500))",
                             ("from scipy.linalg import svd;"
                              "from numpy.random import randn"), number=n)
        fast = timeit.timeit("randomized_svd(randn(500, 500))",
                             ("from randomized_svd import randomized_svd;"
                              "from numpy.random import randn"), number=n)
        self.assertLess(fast, full / 2.0)

    @unittest.skip('not implemented')
    def test_adaptive_range_small_error(self):
        A = np.random.randn(50, 50)
        Q = randomized_svd.adaptive_range(A, 1e-9)
        err = la.norm((np.eye(50) - Q.dot(Q.T)).dot(A), 2)
        self.assertLess(err, 1e-9)
        self.assertGreater(err, 0)
