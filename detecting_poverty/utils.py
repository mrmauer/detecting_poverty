import torch
import numpy as np
from PIL import Image
import random

class LatLonBBDraw():
    '''
    A callable class to provide random samples of fixed size bounding
    boxes from within a larger (super) bounding box.
    '''
    def __init__(self, lat_range, lon_range, dimension):
        """
        Create parameters for sampling function.

        Inputs:
            lat_range (tuple of two floats): north and south boundaries
                of super bounding box.
            lon_range (tuple of two floats): west and east boundaries.
            dimension (float): width (and height) in degrees of desired bounding
                box sub samples.
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.dimension = dimension
    def __call__(self):
        """
        Returns:
            A length 4 tuple of floats: (west, south, east, north)
        """
        lat = random.uniform(self.lat_range[0], self.lat_range[1])
        lon = random.uniform(self.lon_range[0], self.lon_range[1])
        return (lon, lat, lon+self.dimension, lat+self.dimension)

class LatLonDraw():
    '''
    A callable class to provide random samples of lat/lon pairs 
    from within a bounding box.
    '''
    def __init__(self, lat_range, lon_range):
        """
        Create parameters for sampling function.

        Inputs:
            lat_range (tuple of two floats): north and south boundaries
                of super bounding box.
            lon_range (tuple of two floats): west and east boundaries.
        """
        self.lat_range = lat_range
        self.lon_range = lon_range
    def __call__(self):
        """
        Returns:
            A length 2 tuple of floats: (lat, lon)
        """
        lat = random.uniform(self.lat_range[0], self.lat_range[1])
        lon = random.uniform(self.lon_range[0], self.lon_range[1])
        return (lon, lat)

def resize(digits, row_size, column_size):
    """
    Resize images from input scale to row-size x clumn_size
    @row_size,column_size : scale_size intended to be
    """

    return np.array(
        [
            np.array(Image.fromarray(_).resize((row_size, column_size)))
            for _ in digits
        ]
    )


def gen_solution(test_lst, fname):
    """
    Generate csv file for Kaggle submission
    ------
    :in:
    test_lst: 1d array of (n_data), predicted test labels
    fname: string, name of output file
    """

    heads = ['Id', 'Category']
    with open(fname, 'w') as fo:
        fo.write(heads[0] + ',' + heads[1] + '\n')
        for ind in range(len(test_lst)):
            fo.write(heads[0] + ' ' + str(ind + 1) + ',' + str(test_lst[ind]) + '\n')
            

def x2p(X, tolerance=1e-5, perplexity=30.0, max_steps=50, verbose=True):
    """
    Binary search for sigma of conditional Gaussians, used in t-SNE
    ------
    :in:
    X: 2d array of shape (n_data, n_dim), data matrix
    tolerance: float, maximum difference between perplexity and desired perplexity
    max_steps: int, maximum number of binary search steps
    verbose: bool, 
    :out:
    P: 2d array of shape (n_data, n_dim), Probability of conditional Gaussian P_i|j
    """

    n, d = X.shape

    X2 = np.sum(X**2, axis=1, keepdims=True)
    D = X2 - 2*np.matmul(X, X.T) + X2.T
    P = np.zeros((n, n))
    betas, sum_beta = np.ones(n), 0
    desired_entropy = np.log(perplexity)

    for i in range(n):

        if i % 1000 == 0 and verbose:
            print("Computing Gaussian for point %d of %d..." % (i, n))

        betamin, betamax = -np.inf, np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        beta = 1.0
        for s in range(max_steps):
            Pi = np.exp(-Di*beta)
            sum_Pi = Pi.sum()
            entropy = np.log(sum_Pi) + np.sum(Pi*Di*beta)/sum_Pi
            Pi = Pi/sum_Pi
            diff = entropy - desired_entropy
            if abs(diff) > tolerance:
                if diff > 0:
                    betamin = beta
                    beta = 2.*beta if betamax == np.inf else (beta+betamax)/2.
                else:
                    betamax = beta
                    beta = beta/2. if betamin == -np.inf else (beta+betamin)/2.
            else:
                break
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pi
        sum_beta += beta
    if verbose:
        print("Gaussian for %d points done" % (n))
    return P


class JacobOptimizer(object):
    """
    Optimizer used in original t-SNE paper
    """

    def __init__(self, parameters, lr, momentum=0.5, min_gain=0.01):
        self.p = next(parameters)
        self.lr = lr
        self.momentum = momentum
        self.min_gain = min_gain
        self.update = torch.zeros_like(self.p)
        self.gains = torch.ones_like(self.p)
        self.iter = 0

    def step(self):
        inc = self.update * self.p.grad < 0.0
        dec = ~inc
        self.gains[inc] += 0.2
        self.gains[dec] *= 0.8
        self.gains.clamp_(min=self.min_gain)
        self.update = self.momentum * self.update - self.lr * self.gains * self.p.grad
        self.p.data.add_(self.update)
        self.iter += 1
        if self.iter >= 250:
            self.momentum = 0.8
