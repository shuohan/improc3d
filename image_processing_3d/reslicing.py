# -*- coding: utf-8 -*-

"""Functions to reslice a 3D image.

The package provides two types of reslicing. The first one uses interpolation
transform the image. See :func:`reslice3d`. The second one permutes and/or
flips the axes of the image to perform the affine, assuming the affine matrix is
close to a permutation matrix with flipping (only a single entry of each column
and row is close to 1 or -1 while others are approximately 0). See
:func:`reslice3d_coarse`.

NOTE:
    This package define the :red:`axial` coordinate as the :red:`RAI minus`
    coordinate, the :green:`coronal` cooridnate as the :green:`RSA minus`
    coordinate, and the :blue:`sagittal` coordinate as the :blue:`ASR minus`
    coordinate. See :func:`transform_to_axial`, :func:`transform_to_coronal`,
    and :func:`transform_to_sagittal`.

Check `nibabel coordinate system
<https://nipy.org/nibabel/coordinate_systems.html>`_ and this
`page <http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm>`_
for more information on the coordinates. `nibabel
<https://nipy.org/nibabel>`_ can calculate the LPI minus affine matrix when
loading a NIfTI image.

Since the coarse reslicing is much faster than using interpolation, the
following can be used:

>>> image_LPIm = reslice3d(image, LPIm_affine) # transform the image into LPI-
>>> image_axial = transform_to_axial(image_LPIm, np.eye(4), coarse=True)
>>> image_coronal = transform_to_coronal(image_LPIm, np.eye(4), coarse=True)
>>> image_sagittal = transform_to_sagittal(image_LPIm, np.eye(4), coarse=True)

"""

import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .utils import convert_grid_to_coords, calc_image_coords
from .utils import convert_points_to_homogeneous
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


def reslice3d(image, affine, order=1, target_shape=None, pivot_point=None):
    """Transforms a 3D image using an affine matrix with interpolation.

    Args:
        image (numpy.ndarray): The image to transform.
        affine (numpy.ndarray): 4x4 affine matrix to transform the image.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.
        target_shape (tuple, optional): 3 :class:`int` spatial shape of the
            transformed image. If ``None``, use the transformed corners from
            the source image to calculate the targe ranage; otherwise, the
            target shape is symmetric around the transformed ``pivot_point``
            from the source image.
        pivot_point (tuple, optional): :class:`int` 3D point for calculating
            the target range with ``target_shape``. If ``None``, use the center
            of the source image.

    Returns:
        numpy.ndarray: The transformed image.

    """
    if target_shape is None:
        target_range = _calc_target_coords_range(image.shape, affine)
        target_shape = (target_range[:, 1] - target_range[:, 0]).astype(int)
    else:
        target_shape_r = np.reshape(target_shape, (3, 1))
        if pivot_point is None:
            pivot_point = (np.array(image.shape[-3:]) - 1) / 2.0
        pivot_point = np.reshape(pivot_point, (3, 1))
        pivot_point = convert_points_to_homogeneous(pivot_point)
        t_pivot = (affine @ pivot_point)[:3, :]
        starts = t_pivot - target_shape_r / 2.0
        stops = starts + target_shape_r
        target_range = np.hstack((starts, stops))
    target_coords = calc_image_coords(target_range)
    affine_t2s = np.linalg.inv(affine)
    source_coords = affine_t2s @ target_coords
    result = map_coordinates(image, source_coords[:3, :], order=order)
    result = np.reshape(result, target_shape)
    return result


def reslice3d_coarse(image, affine):
    """Transforms a 3D image using affine matrix with flipping and permutation.

    Assuming a single entry of each column and row of the affine matrix is
    dominant (i.e. close to a permutation matrix with reflection), this funciton
    flips and permutes the axes of the input image to roughly transform the
    image into the target coordinate. It is faster than :func:`reslice3d` since
    it does not do interpolation.

    Args:
        image (numpy.ndarray): The image to transform.
        affine (numpy.ndarray): 4x4 affine matrix to transform the image.

    Returns:
        numpy.ndarray: The transformed image.

    """
    print(affine)
    axes = np.argmax(np.abs(affine[:3, :3]), axis=1)
    print(axes)
    result = np.transpose(image, axes=axes)
    axes = np.arange(3)[np.sum(affine[:3, :3], axis=1)<0]
    print(axes)
    result = np.flip(result, axis=axes)
    print('---')
    return result


def _calc_target_coords_range(source_shape, affine, round=True):
    """Calculates the range of the target coordinates.

    Args:
        source_shape (tuple): 3 :class:`int` spatial shape of the image.
        affine (numpy.ndarray): 4x4 affine matrix.
        round (bool, optional): Floor the min and ceil the max if ``True``

    Returns:
        numpy.ndarray: 3x2 matrix of the target coordinate ranges for the 3
            axes. The first column is the range starts and the second column is
            the range stops (the largest coordinates + 1).

    """
    grid = np.meshgrid(*[(0, dim-1) for dim in source_shape])
    coords = convert_grid_to_coords(grid)
    coords = convert_points_to_homogeneous(coords)
    coords = (affine @ coords)[:3]
    starts = np.min(coords, axis=1)[..., None]
    stops = np.max(coords, axis=1)[..., None] + 1
    if round:
        starts = np.floor(starts)
        stops = np.ceil(stops)
    return np.hstack((starts, stops))


def transform_to_axial(image, LPIm_affine, order=1, coarse=False):
    """Transforms the 3D image into the axial view.

    Args:
        image (numpy.ndarray): The image to transform.
        LPIm_affine (numpy.ndarray): 4x4 affine matrix.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.
        coarse (bool, optional): Use :func:`reslice3d_coarse` if ``True``.
            Use :func:`reslice3d` otherwise.

    Returns:
        numpy.ndarray: The transformed image.

    """
    affine = convert_LPIm_to_RAIm(LPIm_affine)
    if coarse:
        return reslice3d_coarse(image, affine)
    else:
        return reslice3d(image, affine, order=order)


def transform_to_coronal(image, LPIm_affine, order=1, coarse=False):
    """Transforms the 3D image into the coronal view.

    Args:
        image (numpy.ndarray): The image to transform.
        LPIm_affine (numpy.ndarray): 4x4 affine matrix.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.
        coarse (bool, optional): Use :func:`reslice3d_coarse` if ``True``.
            Use :func:`reslice3d` otherwise.

    Returns:
        numpy.ndarray: The transformed image.

    """
    affine = convert_LPIm_to_RSAm(LPIm_affine)
    if coarse:
        return reslice3d_coarse(image, affine)
    else:
        return reslice3d(image, affine, order=order)


def transform_to_sagittal(image, LPIm_affine, order=1, coarse=False):
    """Transforms the 3D image into the sagittal view.

    Args:
        image (numpy.ndarray): The image to transform.
        LPIm_affine (numpy.ndarray): 4x4 affine matrix.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.
        coarse (bool, optional): Use :func:`reslice3d_coarse` if ``True``.
            Use :func:`reslice3d` otherwise.

    Returns:
        numpy.ndarray: The transformed image.

    """
    affine = convert_LPIm_to_ASRm(LPIm_affine)
    if coarse:
        return reslice3d_coarse(image, affine)
    else:
        return reslice3d(image, affine, order=order)


def calc_transformed_shape(image_shape, affine):
    """Calculates the shape of the transformed 3D image by an affine matrix.
    
    Args:
        image_shape (tuple): 3 :class:`int` spatial shape of the image.
        affine (numpy.ndarray): 4x4 affine matrix.

    Returns:
        tuple: 3 :class:`int` spatial shape of the transformed image.

    """
    target_range = _calc_target_coords_range(image_shape, affine, round=True)
    target_shape = (target_range[:, 1] - target_range[:, 0]).astype(int)
    return tuple(target_shape)
