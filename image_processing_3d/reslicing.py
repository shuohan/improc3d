# -*- coding: utf-8 -*-

"""Functions to reslice a 3D image.

Check `nibabel <https://nipy.org/nibabel/coordinate_systems.html>`_ and this
`page <http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm>_` for
more information on the coordinates. This package define the axial coordinate as
the RAI minus coordinate, the coronal cooridnate as the RSA minus coordinate,
and the sagittal coordinate as the ASR minus coordinate.

`nibabel <https://nipy.org/nibabel>`_ can calculate the LPI minus affine matrix
when loading a NIfTI image.

"""

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .utils import convert_grid_to_coords, convert_points_to_homogeneous
from .utils import convert_translation_to_homogeneous


def convert_LPIm_to_RAIm(LPIm_affine):
    """Converts a LPI minus affine matrix into RAI minus for a 3D image.

    Args:
        LPIm_affine (numpy.ndarray): The 4x4 affine matrix to transform the
        image array into LPI minus coordinates.
    
    Returns:
        numpy.ndarray: The affine matrix to transform the image array into RAI
            minus coordinates.

    """
    LPI_to_RAI_affine = np.array([[-1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    affine = LPI_to_RAI_affine @ LPIm_affine
    return affine


def convert_LPIm_to_RSAm(LPIm_affine):
    """Converts a LPI minus affine matrix into RSA minus for a 3D image.

    Args:
        LPIm_affine (numpy.ndarray): The 4x4 affine matrix to transform the
            image array into LPI minus coordinates.
    
    Returns:
        numpy.ndarray: The affine matrix to transform the image array into RSA 
            minus coordinates.

    """
    LPI_to_RSA_affine = np.array([[-1, 0, 0, 0],
                                  [0, 0, -1, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 1]])
    affine = LPI_to_RSA_affine @ LPIm_affine
    return affine


def convert_LPIm_to_ASRm(LPIm_affine):
    """Converts a LPI minus affine matrix into ASR  minus for a 3D image.

    Args:
        LPIm_affine (numpy.ndarray): The 4x4 affine matrix to transform the
            image array into LPI minus coordinates.
    
    Returns:
        numpy.ndarray: The affine matrix to transform the image array into ASR
            minus coordinates.

    """
    LPI_to_ASR_affine = np.array([[0, -1, 0, 0],
                                  [0, 0, -1, 0],
                                  [-1, 0, 0, 0],
                                  [0, 0, 0, 1]])
    affine = LPI_to_ASR_affine @ LPIm_affine
    return affine


def reslice(image, affine, order=1):
    """Transforms a 3D image using an affine matrix

    Args:
        image (numpy.ndarray): The image to transform.
        affine (numpy.ndarray): 4x4 affine matrix to transform the image.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.

    Returns:
        numpy.ndarray: The transformed image.

    """
    target_range = _calc_target_coords_range(image.shape, affine)
    target_shape = _calc_target_shape(target_range)
    grid = np.meshgrid(*[np.arange(s) for s in target_shape], indexing='xy')
    target_coords = convert_grid_to_coords(grid)
    target_coords = convert_points_to_homogeneous(target_coords)

    offset = convert_translation_to_homogeneous(target_range[:, 0])
    affine_t2s = np.linalg.inv(affine) @ offset
    source_coords = affine_t2s @ target_coords
    result = map_coordinates(image, source_coords[:3, :], order=order)
    result = np.reshape(result, grid[0].shape)
    return result


def _calc_target_coords_range(source_shape, affine):
    """Calculates the range of the target coordinates.

    Args:
        source_shape (tuple): 3 :class:`int` spatial shape of the image.
        affine (numpy.ndarray): 4x4 affine matrix.

    Returns:
        numpy.ndarray: 3x2 matrix of the target coordinate ranges for the 3
            axes. The first column is the lower bound and the second column is
            the upper bound.

    """
    grid = np.meshgrid(*[(0, dim-1) for dim in source_shape])
    coords = convert_grid_to_coords(grid)
    coords = convert_points_to_homogeneous(coords)
    coords = (affine @ coords)[:3]
    min_coords = np.min(coords, axis=1)[..., None]
    max_coords = np.max(coords, axis=1)[..., None]
    return np.hstack((min_coords, max_coords))


def _calc_target_shape(target_range):
    """Returns the shape of the target image"""
    return np.ceil(target_range[:, 1]-target_range[:, 0]+1).astype(int)


def transform_to_axial(image, LPIm_affine, order=1):
    """Transforms the 3D image into the axial view.

    Args:
        image (numpy.ndarray): The image to transform.
        LPIm_affine (numpy.ndarray): 4x4 affine matrix.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.

    Returns:
        numpy.ndarray: The transformed image.

    """
    affine = convert_LPIm_to_RAIm(LPIm_affine)
    return reslice(image, affine, order=order)


def transform_to_coronal(image, LPIm_affine, order=1):
    """Transforms the 3D image into the coronal view.

    Args:
        image (numpy.ndarray): The image to transform.
        LPIm_affine (numpy.ndarray): 4x4 affine matrix.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.

    Returns:
        numpy.ndarray: The transformed image.

    """
    affine = convert_LPIm_to_RSAm(LPIm_affine)
    return reslice(image, affine, order=order)


def transform_to_sagittal(image, LPIm_affine, order=1):
    """Transforms the 3D image into the sagittal view.

    Args:
        image (numpy.ndarray): The image to transform.
        LPIm_affine (numpy.ndarray): 4x4 affine matrix.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.

    Returns:
        numpy.ndarray: The transformed image.

    """
    affine = convert_LPIm_to_ASRm(LPIm_affine)
    return reslice(image, affine, order=order)


def calc_transformed_shape(image_shape, affine):
    """Calculates the shape of the transformed 3D image by an affine matrix.
    
    Args:
        image_shape (tuple): 3 :class:`int` spatial shape of the image.
        affine (numpy.ndarray): 4x4 affine matrix.

    Returns:
        tuple: 3 :class:`int` spatial shape of the transformed image.

    """
    target_range = _calc_target_coords_range(image_shape, affine)
    target_shape = _calc_target_shape(target_range)
    return target_shape
