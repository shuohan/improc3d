# -*- coding: utf-8 -*-

import numpy as np


def calc_random_intensity_transform(klim=(5, 20), blim=(-1, 1), num_sigmoid=5):
    """Calculate random intensity transform

    Uniformly sample k, b, alpha for mixture of sigmoid to calculate a intensity
    transform

    Args:
        klim (tuple of float): The lower and higher bound of k
        blim (tuple of float): The lower and higher bound of b
        num_sigmoid (int): The number of sigmoid for the mixture

    Returns:
        transform (function): The function to map the intensity

    """
    ks = _sample_k(*klim, num_sigmoid)
    bs = _sample_b(*blim, num_sigmoid)
    alphas = _sample_alpha(num_sigmoid)
    def transform(x, ks=ks, bs=bs, alpha=alphas):
        low, high = -0.5, 0.5
        orig_x_min, orig_x_max = np.min(x), np.max(x)
        y = _scale(x, low, high)
        y = _mixture_sigmoid(y, ks, bs, alphas)
        y = _scale(y, orig_x_min, orig_x_max)
        return y
    return transform


def _sample_k(k_min, k_max, size):
    """Sample k from a uniform distribution

    k can enlarge or shrink the data. Sample k so the largest possible value is
    `k_max` and the smallest value is `k_min` when enlarging; shrinking is 1
    divided by the enarging factor

    Args:
        k_max (float): The largest enlarging factor
        k_min (float): The smallest enlarging factor
        size (int): The number of sampled k

    Returns:
        ks (np.array): The sampled k

    """
    ks = np.random.rand(size) * (k_max / k_min - 1) + 1 # from [1, k_max/k_min]
    shrinking_indices  = np.random.choice((-1, 1), size=size) < 0
    ks[shrinking_indices] = 1 / ks[shrinking_indices] # shrinking or enlarging
    ks = ks * k_max / k_min
    return ks


def _sample_b(b_min, b_max, size):
    """Sample b from a uniform distribution [`b_min`, `b_max`]

    Args:
        b_min (float): The smallest value
        b_max (float): The largest value
        size (int): The number of sampled b

    Returns:
        bs (np.array): The sampled b

    """
    bs = np.random.rand(size) * (b_max - b_min) + b_min
    return bs


def _sample_alpha(size):
    """Sample a from a uniform distribution [0, 1]

    Args:
        size (int): The number of sampled alphas

    Returns:
        alphas (np.array): The sampled alphas

    """
    return np.random.rand(size)


def _sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def _mixture_sigmoid(x, ks, bs, alphas):
    assert len(ks) == len(bs) == len(alphas)
    alphas = np.array(alphas) / np.sum(alphas)
    y = list()
    for k, b, a in zip(ks, bs, alphas):
        y.append(a * _sigmoid(k * (x + b)))
    y = np.sum(y, axis=0)
    return y

    
def _scale(x, low, high):
    """Scale `x` so its min is `low` and max is `high`
    
    Args:
        x (numpy.array): The array to scale
        low (float): The target minimum value
        high (float): The target maximum value

    Returns:
        y (numpy.array): The scaled array

    """
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = y * (high - low) + low
    return y
