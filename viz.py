import numpy as np
import scipy.linalg as la
import scipy.linalg.interpolative as sli
import matplotlib.pyplot as pt
import scipy
import time
import numpy.linalg as nla
import math


def pca_montage(C, imsize=None, grid_width=None):
    nimages = C.shape[0]
    if grid_width is None:
        grid_width = int(math.ceil(math.sqrt(nimages)))

    if imsize is None:
        imsize = (int(math.ceil(math.sqrt(C.shape[1]))),
                  int(math.floor(math.sqrt(C.shape[1]))))

    row = []
    rows = None
    for i in range(nimages):
        im = C[i, :].reshape(imsize)
        row.append(im)
        if len(row) >= grid_width:
            if rows is None:
                rows = np.hstack(tuple(row))
            else:
                rows = np.vstack((rows, np.hstack(tuple(row))))
            row = []


    if len(row) > 0:
        for i in range(grid_width-len(row)):
            row.append(np.zeros(imsize))

        rows = np.vstack((rows, np.hstack(tuple(row))))

    return rows
