# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from .utils import convert_grid_to_coords


def deform(image, x_deformation, y_deformation, z_deformation, order=1):
    """Transform an image using deformation field
    
    Args:
        image (3d numpy.array): The 3d image to deform
        x_deformation (3d numpy.array): The deformation along x axis in pixel
        y_deformation (3d numpy.array): The deformation along y axis in pixel
        z_deformation (3d numpy.array): The deformation along z axis in pixel
        order (int): The interpolation order

    Returns:
        deformed_image (3d numpy.array): The deformed image

    """
    shape = image.shape
    target_grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    deformation = [x_deformation, y_deformation, z_deformation]
    source_grid = [g - d for g, d in zip(target_grid, deformation)]
    source_coords = convert_grid_to_coords(source_grid)
    interpolation = map_coordinates(image, source_coords, order=order)
    deformed_image = np.reshape(interpolation, shape)
    return deformed_image


def calc_random_defromation(image_shape, sigma, scale):
    """Calculate a component of a random deformation field

    This deformation is along one axis. Call this function three times from
    deformation along x, y, and z

    Args:
        image_shape ((3,) tuple): The shape of the image
        sigma (float): The value controling the smoothness of the deformation
            field. Larger the value is, smoother the field.
        scale (float): The deformation is supposed to draw from a uniform
            distribution [-eps, +eps]. Use this value to specify the upper bound
            of the sampling distribution.

    Returns:
        result (3d numpy.array): The component of the deformation filed 

    """
    random_state = np.random.RandomState(None)
    result = random_state.rand(*image_shape) * 2 - 1
    result = gaussian_filter(result, sigma)
    result = result / np.max(result) * scale
    return result
