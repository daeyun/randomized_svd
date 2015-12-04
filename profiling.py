import randomized_svd
import test_randomized_svd

A = test_randomized_svd.lowrank(1000, 1000)

randomized_svd.randomized_svd(A, k=200)