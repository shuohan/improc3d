# -*- coding: utf-8 -*-

import numpy as np


def convert_grid_to_coords(grid):
    """Convert meshgrid to coordinate points

    Convert the meshgrid, which is the indices of the image pixels along x, y,
    and z axes, to a matrix, which is stacked coordinate vectors.

    For example, it convert 

    [[0, 0],    [[0, 1],
     [1, 1]]     [0, 1]]

    to

    [[0, 0, 1, 1]
     [0, 1, 0, 1]]
    
    Args:
        grid ((num_dims,) tuple of numpy.array): Meshgrid of image pixel indices

    Returns:
        coords (num_dims x num_pixels numpy.array): The coordinate vectors

    """
    coords = np.vstack([g.flatten()[None, ...] for g in grid]) 
    return coords
