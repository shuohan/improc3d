# -*- coding: utf-8 -*-

"""Homogeneous conversions

"""

import numpy as np


def convert_rotation_to_homogeneous(rotation):
    """Convert the 3x3 rotation matrix to homogeneous coordinate

    Args:
        rotation (3x3 numpy.array): The rotation matrix to convert
    
    Returns:
        result (4x4 numpy.array): Homogeneous rotation matrix

    """
    result = np.eye(4)
    result[:3, :3] = rotation
    return result


def convert_translation_to_homogeneous(translation):
    """Convert 3D translation to homogeneous coordinate

    Args:
        translation ((3,) numpy.array or list): The translation to convert

    Returns:
        result (4x4 numpy.array): Translation in homogeneous coordinate

    """
    result = np.eye(4)
    result[:3, 3] = translation
    return result


def convert_points_to_homogeneous(points):
    """Convert 3D points to homogeneous coordinate

    Args:
        points (3 x num_points numpy.array): The points to convert

    Returns:
        points (4 x num_points numpy.array): Points in homogeneous coordinate

    """
    points = np.vstack([points, np.ones((1, points.shape[1]))])
    return points


def convert_points_from_homogeneous(points):
    """Convert 3D points from homogeneous coordinate

    Args:
        points (4 x num_points numpy.array): The points to convert

    Returns:
        points (3 x num_points numpy.array): Non-homogeneous points

    """
    return points[:3, :]
