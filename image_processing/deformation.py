# -*- coding: utf-8 -*-

import numpy as np


def deform(image, x_deformation, y_deformation, z_deformation, order=order):
    shape = image.shape
    target_grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    deformation = [x_deformation, y_deformation, z_deformation]
    source_grid = [g - d for g, d in zip(target_grid, deformation)]
    source_coords = convert_grid_to_coords(source_grid)
    interpolation = map_coordinates(image, source_coords, order=order)
    deformed_image = np.reshape(interpolation, image.shape)
    return deformed_image


def calc_random_defromation(image_shape, sigma, scale_limit):
    random_state = np.random.RandomState(None)
    max_scale = random_state.rand(1) * scale_limit
    min_scale = -random_state.rand(1) * scale_limit
    result = random_state.rand(*image_shape)
    result = gaussian_filter(result, sigma)
    result = smoothed_deformation / np.max(result)
    result = scaled_deformation * (max_scale - min_scale) - min_scale
    return result
