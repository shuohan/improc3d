# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .reslicing import reslice3d
from .utils import convert_translation_to_homogeneous


def rotate3d(image, x_angle, y_angle, z_angle, point=None, order=1):
    """Rotates an 3D image around a point.

    This function assumes 0 outside the image. If the input image is 4D, it
    assumes channel-first and applies the same rotation for each of the
    channels.
    
    Args:
        image (numpy.ndarray): The 3D or 4D image to rotate; channel first when
            it is 4D.
        x_angle (float): Rotation angle around x axis in degrees.
        y_angle (float): Rotation angle around y axis in degrees.
        z_angle (float): Rotation angle around z axis in degrees.
        point (tuple, optional): The 3D rotation point; use the image center as
            the rotation point if ``None``. Otherwise, it can be :class:`tuple`
            or :class:`numpy.ndarray` of :class:`float`.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.

    """
    # if point is None:
    #     point = np.array(image.shape[-3:]) / 2
    # translation = convert_translation_to_homogeneous(-np.array(point))
    
    rotation_x = _calc_rotation_x(x_angle / 180 * np.pi)
    rotation_y = _calc_rotation_y(y_angle / 180 * np.pi)
    rotation_z = _calc_rotation_z(z_angle / 180 * np.pi)
    rotation = rotation_z @ rotation_y @ rotation_x

    return reslice3d(image, rotation)
    # if len(image.shape) == 4:
    #     interpolation = [map_coordinates(im, source_coords, order=order)
    #                      for im in image]
    #     interpolation = np.vstack(interpolation)
    # else:
    #     interpolation = map_coordinates(image, source_coords, order=order)
    # rotated_image = np.reshape(interpolation, image.shape)


def _calc_rotation_x(angle):
    """Calculates 3D rotation matrix around the first axis.

    Args:
        angle (float): The rotation angle in rad.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.

    """
    rotation = np.array([[1, 0, 0, 0],
                         [0, np.cos(angle), -np.sin(angle), 0],
                         [0, np.sin(angle), np.cos(angle), 0],
                         [0, 0, 0, 1]])
    return rotation


def _calc_rotation_y(angle):
    """Calculates 3D rotation matrix around the second axis.

    Args:
        angle (float): The rotation angle in rad.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.

    """
    rotation = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                         [0, 1, 0, 0],
                         [-np.sin(angle), 0, np.cos(angle), 0],
                         [0, 0, 0, 1]])
    return rotation


def _calc_rotation_z(angle):
    """Calculates 3D rotation matrix around the third axis.

    Args:
        angle (float): The rotation angle in rad.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.

    """
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
                         [np.sin(angle), np.cos(angle), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
    return rotation
