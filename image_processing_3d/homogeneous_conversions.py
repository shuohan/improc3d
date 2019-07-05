# -*- coding: utf-8 -*-

"""Functions for homogeneous conversions.

"""

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
