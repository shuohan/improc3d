# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .utils import convert_grid_to_coords
from .homogeneous_conversions import convert_rotation_to_homogeneous
from .homogeneous_conversions import convert_translation_to_homogeneous
from .homogeneous_conversions import convert_points_to_homogeneous
from .homogeneous_conversions import convert_points_from_homogeneous


def rotate3d(image, x_angle, y_angle, z_angle, point=None, order=1):
    """Rotates an 3D image around a point.
    
    Args:
        image (numpy.ndarray): The 3D or 4D image to rotate; channel first when
            it is 4D.
        x_angle (float): Rotation angle around x axis in degrees.
        y_angle (float): Rotation angle around y axis in degrees.
        z_angle (float): Rotation angle around z axis in degrees.
        point (tuple, optional): The 3D rotation point; use the image center as
            the rotation point if ``None``. Otherwise, it can be :class:`tuple`
            or :class:`numpy.ndarray` of :class:`float`.
        order (int, optional): The interpolation order.

    """
    if point is None:
        point = np.array(image.shape[-3:]) / 2
    
    rotation_x = _calc_rotation_x(x_angle / 180 * np.pi)
    rotation_y = _calc_rotation_y(y_angle / 180 * np.pi)
    rotation_z = _calc_rotation_z(z_angle / 180 * np.pi)
    rotation = rotation_z @ rotation_y @ rotation_x

    inverse_rotation = np.linalg.inv(rotation)
    inverse_transform = _calc_rotation_around_point(inverse_rotation, point)

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
    rotated_image = np.reshape(interpolation, image.shape)

    return rotated_image 


def _calc_rotation_around_point(rotation, point):
    """Calculates the rotation around a point.

    It first translates the image so the ``point`` is at the origin, then
    applies the rotation, and finally translates the image back so ``point``
    does not change.
    
    Args:
        rotation (numpy.ndarray): The 3x3 rotation matrix.
        point (numpy.ndarray): The 3D rotation point.

    Returns:
        numpy.ndarray: The 4x4 homogeneous rotation matrix.

    """
    rotation = convert_rotation_to_homogeneous(rotation)
    shift_to_origin = convert_translation_to_homogeneous(-point)
    shift_back = convert_translation_to_homogeneous(point)
    transformation = shift_back @ rotation @ shift_to_origin
    return transformation


def _calc_rotation_x(angle):
    """Calculates 3D rotation matrix around the first axis.

    Args:
        angle (float): The rotation angle in rad.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.

    """
    rotation = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    return rotation


def _calc_rotation_y(angle):
    """Calculates 3D rotation matrix around the second axis.

    Args:
        angle (float): The rotation angle in rad.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.

    """
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    return rotation


def _calc_rotation_z(angle):
    """Calculates 3D rotation matrix around the third axis.

    Args:
        angle (float): The rotation angle in rad.
    
    Returns:
        numpy.ndarray: The 3x3 rotation matrix.

    """
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    return rotation


def calc_image_coords(shape):
    """Calculates the coordinates of all image voxels.

    Args:
        shape (tuple): The 3-element :py:class:`int` spatial shape of the image.

    Returns:
        coords (numpy.ndarray): The coordinates of image voxels

    """
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    coords = convert_grid_to_coords(grid)
    return coords
