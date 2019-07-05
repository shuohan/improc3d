# -*- coding: utf-8 -*-

import numpy as np

from .homogeneous_conversions import convert_transformation_to_homogeneous
from .homogeneous_conversions import convert_translation_to_homogeneous


def convert_grid_to_coords(grid):
    """Converts a meshgrid to coordinate points.

    This function converts the meshgrid which is the indices of the image pixels
    along x, y, and z axes, to a matrix which is stacked coordinate vectors.

    For example, it converts::
    
        [[1, 2],    [[5, 6],
         [3, 4]]     [7, 8]]

    to::

        [[1, 2, 3, 4]
         [5, 6, 7, 8]]
    
    Args:
        grid (tuple): The meshgrid of image voxel indices. The number of element
            is equal to the dimention of the image and each element is the
            :class:`numpy.ndarray` coordinates along an axis.

    Returns:
        numpy.ndarray: The num_dims x num_pixels coordinate vectors.

    """
    coords = np.vstack([g.flatten()[None, ...] for g in grid]) 
    return coords


def calc_image_coords(shape):
    """Calculates the coordinates of all image voxels.

    Args:
        shape (tuple): The 3-element :py:class:`int` spatial shape of the image.

    Returns:
        numpy.ndarray: The num_dims x num_pixels coordinate vectors.

    """
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    coords = convert_grid_to_coords(grid)
    return coords


def calc_transformation_around_point(trans, point):
    """Calculates the transformation around a point.

    It first translates the image so the ``point`` is at the origin, then
    applies the rotation, scaling, etc., and finally translates the image back
    so ``point`` does not change.
    
    Args:
        trans (numpy.ndarray): The 3x3 matrix to transform a 3D point.
        point (numpy.ndarray): The 3D rotation point.

    Returns:
        numpy.ndarray: The 4x4 homogeneous transformation matrix.

    """
    trans = convert_transformation_to_homogeneous(trans)
    shift_to_origin = convert_translation_to_homogeneous(-point)
    shift_back = convert_translation_to_homogeneous(point)
    trans = shift_back @ trans @ shift_to_origin
    return trans
