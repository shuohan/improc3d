# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .homogeneous_conversions import convert_rotation_to_homogeneous
from .homogeneous_conversions import convert_translation_to_homogeneous
from .homogeneous_conversions import convert_points_to_homogeneous
from .homogeneous_conversions import convert_points_from_homogeneous


def rotate3d(image, x_angle, y_angle, z_angle, point=None, order=1):
    """Rotate a 3D image around a point
    
    Args:
        image (3D numpy.array): The image to rotate
        x_angle (float): Rotation angle around x axis in degree
        y_angle (float): Rotation angle around y axis in degree
        z_angle (float): Rotation angle around z axis in degree
        point ((3,) tuple): The rotation point; if None, use the image center
        order (int): The interpolation order

    """
    if not point:
        point = np.array(image.shape) / 2
    
    rotation_x = _calc_rotation_x(x_angle / 180 * np.pi)
    rotation_y = _calc_rotation_y(y_angle / 180 * np.pi)
    rotation_z = _calc_rotation_z(z_angle / 180 * np.pi)
    rotation = rotation_z @ rotation_y @ rotation_x

    inverse_rotation = np.linalg.inv(rotation)
    inverse_transform = _calc_rotation_around_point(inverse_rotation, point)

    target_coords = calc_image_coords(image.shape)
    target_coords = convert_points_to_homogeneous(target_coords)

    source_coords = inverse_transform @ target_coords
    source_coords = convert_points_from_homogeneous(source_coords)

    interpolation = map_coordinates(image, source_coords, order=order)
    rotated_image = np.reshape(interpolation, image.shape)

    return rotated_image 


def _calc_rotation_around_point(rotation, point):
    """Calculate the rotation around a point

    It first translates the image so the `point` is at the origin, then
    applies the rotation, and finally translates the image back so `point`
    does not change.
    
    Args:
        rotation (3x3 numpy.array): The rotation matrix
        point ((3,) numpy.array): The rotation origin

    Returns:
        transformation (4x4 numpy.array): Homogeneous rotation

    """
    rotation = convert_rotation_to_homogeneous(rotation)
    shift_to_origin = convert_translation_to_homogeneous(-point)
    shift_back = convert_translation_to_homogeneous(point)
    transformation = shift_back @ rotation @ shift_to_origin
    return transformation


def _calc_rotation_x(angle):
    """Calculate 3D rotation matrix around the first axis

    Args:
        angle (float): The rotation angle in rad
    
    Returns:
        rotation (3x3 numpy.array): The rotation matrix

    """
    rotation = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    return rotation


def _calc_rotation_y(angle):
    """Calculate 3D rotation matrix around the second axis

    Args:
        angle (float): The rotation angle in rad
    
    Returns:
        rotation (3x3 numpy array): The rotation matrix

    """
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    return rotation


def _calc_rotation_z(angle):
    """Calculate 3D rotation matrix around the third axis

    Args:
        angle (float): The rotation angle in rad
    
    Returns:
        rotation (3x3 numpy array): The rotation matrix

    """
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    return rotation


def calc_image_coords(shape):
    """Calculate the coordinates of image voxels

    Args:
        shape ((3,) tuple): The shape of the image

    Returns:
        coords (3 x num_points numpy.array): The coordinates of image voxels

    """
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    coords = convert_grid_to_coords(grid)
    return coords


def convert_grid_to_coords(grid):
    coords = np.vstack([g.flatten()[None, ...] for g in grid]) 
    return coords
