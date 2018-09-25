# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .utils import convert_grid_to_coords
from .homogeneous_conversions import convert_rotation_to_homogeneous
from .homogeneous_conversions import convert_translation_to_homogeneous
from .homogeneous_conversions import convert_points_to_homogeneous
from .homogeneous_conversions import convert_points_from_homogeneous
from .rotation import _calc_rotation_around_point, calc_image_coords


_calc_scaling_around_point = _calc_rotation_around_point


def scale3d(image, x_scale, y_scale, z_scale, point=None, order=1):
    """Scale a 3D image around a point

    Scale a 3D image around a point. If the input image is a channel-first
    multi-channel 3D image (i.e. 4D image with the first dimension as channels),
    this funcition scale the image per channel. 
    
    Args:
        image (3D numpy.array): The image to scale
        x_scale (float): Scaling factor along x axis
        y_scale (float): Scaling factor along y axis
        z_scale (float): Scaling factor along z axis
        point ((3,) tuple): The scaling center point. If None, use the image
            center
        order (int): The interpolation order

    Returns:
        scaled_image (3D numpy.array): The scaled image

    """
    if point is None:
        point = np.array(image.shape[-3:]) / 2
    
    inverse_scaling = np.array([[1/x_scale, 0, 0],
                                [0, 1/y_scale, 0],
                                [0, 0, 1/z_scale]])
    inverse_transform = _calc_scaling_around_point(inverse_scaling, point)

    target_coords = calc_image_coords(image.shape[-3:])
    target_coords = convert_points_to_homogeneous(target_coords)

    source_coords = inverse_transform @ target_coords
    source_coords = convert_points_from_homogeneous(source_coords)

    if len(image.shape) == 4:
        interpolation = [map_coordinates(im, source_coords, order=order)
                         for im in image]
        interpolation = np.vstack(interpolation)
    else:
        interpolation = map_coordinates(image, source_coords, order=order)
    scaled_image = np.reshape(interpolation, image.shape)

    return scaled_image 
