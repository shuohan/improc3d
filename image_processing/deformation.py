# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from .utils import convert_grid_to_coords


def deform(image, x_deformation, y_deformation, z_deformation, order=1):
    shape = image.shape
    target_grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    deformation = [x_deformation, y_deformation, z_deformation]
    source_grid = [g - d for g, d in zip(target_grid, deformation)]
    source_coords = convert_grid_to_coords(source_grid)
    interpolation = map_coordinates(image, source_coords, order=order)
    deformed_image = np.reshape(interpolation, image.shape)
    return deformed_image


def calc_random_defromation(image_shape, sigma, scale):
    random_state = np.random.RandomState(None)
    result = random_state.rand(*image_shape) * 2 - 1
    result = gaussian_filter(result, sigma)
    result = result / np.max(result) * scale
    print(np.min(result), np.max(result))
    return result
