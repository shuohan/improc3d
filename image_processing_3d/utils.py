# -*- coding: utf-8 -*-

import numpy as np


def convert_grid_to_coords(grid):
    """Converts a meshgrid to coordinate points.

    This function converts the meshgrid which is the indices of the image pixels
    along x, y, and z axes, to a matrix which is stacked coordinate vectors.

    For example, it converts

    [[1, 2],    [[5, 6],
     [3, 4]]     [7, 8]]

    to

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
