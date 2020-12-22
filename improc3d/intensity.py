import numpy as np


def quantile_scale(image, lower_pct=0.05, upper_pct=0.98,
                   lower_th=0.0, upper_th=1000.0):
    """Scales image intensities using quantiles

    The intensities smaller than or equal to the lower percentage quantile
    ``lower_pct`` will be set to the lower threshold ``lower_th`` and the
    intensities greater than or equal to the upper percentage quantile
    ``upper_pct`` will be set to the upper threshold ``upper_th``. The
    intensities in between will be scaled linearly.

    Args:
        image (numpy.ndarray): The image to scale.
        lower_pct (float, optional): The lower percentage.
        uppder_pct (float, optional): The upper percentage.
        lower_th (float, optional): The lower threshold.
        upper_th (float, optional): The upper threshold.

    Returns:
        numpy.ndarray: The scaled image.

    """
    lower_val = np.quantile(image, lower_pct)
    upper_val = np.quantile(image, upper_pct)
    slope = (upper_th - lower_th) / (upper_val - lower_val)
    intercept = lower_th - slope * lower_val
    scaled = slope * image + intercept
    scaled[scaled>upper_th] = upper_th
    scaled[scaled<lower_th] = lower_th
    return scaled


def calc_random_intensity_transform(klim=(10.0, 20.0), blim=(-1.0, 1.0),
                                    num_sigmoid=5):
    """Calculates a random intensity transform.

    This function uniformly randomly samples :math:`k`, :math:`b`, and
    :math:`\\alpha`` for mixture of sigmoid to create an intensity transform:
    
    .. math::
        y = \\sum_{i}{{\\alpha}_i\\textrm{sigmoid}(k_i(x + b_i))}

    This functions will rescale :math:`{\\alpha}_i` so the sum is 1.

    Args:
        klim (tuple, optional): :class:`float` numbers for the lower and higher
            bound of k.
        blim (tuple, optional): :class:`float` numbers for the lower and higher
            bound of b.
        num_sigmoid (int, optional): The number of sigmoid for the mixture.

    Returns:
        function: The function to map the intensity.

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
    """Samples k from a uniform distribution.

    k can enlarge or shrink the data. This function samples k so the largest
    possible value is ``k_max`` and the smallest value is ``k_min`` when
    enlarging; shrinking is 1 divided by the enarging factor.

    Args:
        k_max (float): The largest enlarging factor.
        k_min (float): The smallest enlarging factor.
        size (int): The number of sampled k.

    Returns:
        numpy.ndarray: The sampled k.

    """
    ks = np.random.rand(size) * (k_max / k_min - 1) + 1 # from [1, k_max/k_min]
    shrinking_indices  = np.random.choice((-1, 1), size=size) < 0
    ks[shrinking_indices] = 1 / ks[shrinking_indices] # shrinking or enlarging
    ks = ks * k_min
    return ks


def _sample_b(b_min, b_max, size):
    """Samples b from a uniform distribution [``b_min``, ``b_max``].

    Args:
        b_min (float): The smallest value.
        b_max (float): The largest value.
        size (int): The number of sampled b.

    Returns:
        numpy.ndarray: The sampled b.

    """
    bs = np.random.rand(size) * (b_max - b_min) + b_min
    return bs


def _sample_alpha(size):
    """Samples alpha from a uniform distribution [0, 1].

    Args:
        size (int): The number of sampled alphas.

    Returns:
        numpy.ndarray: The sampled alphas.

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
    """Scales ``x`` so its min is ``low`` and max is ``high``.
    
    Args:
        x (numpy.ndarray): The array to scale.
        low (float): The target minimum value.
        high (float): The target maximum value.

    Returns:
        numpy.ndarray: The scaled array.

    """
    y = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = y * (high - low) + low
    return y
