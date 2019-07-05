# -*- coding: utf-8 -*-

import numpy as np


def convert_transformation_to_homogeneous(trans):
    """Converts the 3x3 transformation matrix into the homogeneous coordinate.

    Args:
        trans (numpy.ndarray): The 3D transformation matrix to convert.
    
    Returns:
        numpy.ndarray: Homogeneous rotation matrix.

    """
    result = np.eye(4)
    result[:3, :3] = trans
    return result


def convert_translation_to_homogeneous(translation):
    """Converts 3D translation into the homogeneous coordinate.

    Args:
        translation (numpy.ndarray): The 3D translation to convert.

    Returns:
        numpy.ndarray: Translation in the homogeneous coordinate.

    """
    result = np.eye(4)
    result[:3, 3] = translation
    return result


def convert_points_to_homogeneous(points):
    """Converts 3D points into the homogeneous coordinate

    Args:
        points (numpy.ndarray): The points to convert. It should be a 2D array
            with shape 3 x num_points.

    Returns:
        numpy.ndarray: Points in the homogeneous coordinate. A 2D array with
        shape 4 x num_points.

    """
    points = np.vstack([points, np.ones((1, points.shape[1]))])
    return points


def convert_points_from_homogeneous(points):
    """Converts 3D points from the homogeneous coordinate

    Args:
        points (numpy.ndarray): The points to convert. It should a 2D array with
            shape 4 x num_points.

    Returns:
        numpy.ndarray: Non-homogeneous points. A 2D array with shape
        3 x num_points.

    """
    return points[:3, :]


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


def calc_image_coords(shape, indexing='ij'):
    """Calculates the coordinates of all image voxels.

    Args:
        shape (tuple): The 3-element :py:class:`int` spatial shape of the image.
        indexing (str, optional): 'ij' or 'xy'. See :func:`numpy.meshgrid`.

    Returns:
        numpy.ndarray: The num_dims x num_pixels coordinate vectors.

    """
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing=indexing)
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
