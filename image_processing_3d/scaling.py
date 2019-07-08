# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .reslicing import reslice


def scale3d(image, x_scale, y_scale, z_scale, point=None, order=1):
    """Scales a 3D image around a point.

    This function scales a 3D image around a point. If the input image is a
    channel-first multi-channel 3D image (i.e. 4D image with the first dimension
    as channels), this funcition uses the same factors to scale the image for
    each channel.
    
    Args:
        image (numpy.ndarray): The 3D or 4D image to scale. Channel first if 4D.
        x_scale (float): The scaling factor along x axis.
        y_scale (float): The scaling factor along y axis.
        z_scale (float): The scaling factor along z axis.
        point (iterable, optional): The 3D scaling center point. If ``None``,
            use the image center as the scaling center. Otherwise, it can be a
            :py:class:`tuple` or :class:`numpy.ndarray` of :class:float.
        order (int, optional): The interpolation order.

    Returns:
        numpy.ndarray: The 3D or 4D scaled image.

    """
    # if point is None:
    #     point = np.array(image.shape[-3:]) / 2

    scaling = np.array([[x_scale, 0, 0, 0],
                        [0, y_scale, 0, 0],
                        [0, 0, z_scale, 0],
                        [0, 0, 0, 1]])
    return reslice(image, scaling)
