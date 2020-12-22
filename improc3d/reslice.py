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

from .utils import convert_grid_to_coords, calc_image_coords
from .utils import convert_points_to_homogeneous, interp_image


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


def reslice3d(image, affine, order=1, target_shape=None, pivot=None):
    """Transforms a 3D image using an affine matrix with interpolation.

    Args:
        image (numpy.ndarray): The 3D image to transform. Channel first if 4D.
        affine (numpy.ndarray): 4x4 affine matrix to transform the image.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.
        target_shape (tuple, optional): 3 :class:`int` spatial shape of the
            transformed image. If ``None``, use the transformed corners from
            the source image to calculate the targe ranage; otherwise, the
            target shape is symmetric around the transformed ``pivot``
            from the source image.
        pivot (tuple, optional): :class:`int` 3D point for calculating the
            target range with ``target_shape``. If ``None``, use the center of
            the source image.

    Returns:
        numpy.ndarray: The transformed image.

    """
    if target_shape is None:
        target_range = _calc_target_range(image.shape[-3:], affine)
        target_shape = _calc_target_shape(target_range)
    else:
        target_shape = target_shape[-3:]
        pivot = _calc_image_center(image.shape[-3:]) if pivot is None else pivot
        target_range = _calc_target_range_p(target_shape, pivot, affine)
    if len(image.shape) == 4:
        target_shape = (image.shape[0], *target_shape)

    target_coords = calc_image_coords(target_range)
    affine_t2s = np.linalg.inv(affine)
    source_coords = (affine_t2s @ target_coords)[:3, :]
    result = interp_image(image, source_coords, order=order)
    return np.reshape(result, target_shape)


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
    axes = np.argmax(np.abs(affine[:3, :3]), axis=1)
    result = np.transpose(image, axes=axes)
    axes = np.arange(3)[np.sum(affine[:3, :3], axis=1)<0]
    result = np.flip(result, axis=axes)
    return result


def _calc_image_center(image_shape):
    """Returns the center of the image."""
    return (np.array(image_shape[-3:]) - 1) / 2.0

def _calc_target_shape(target_range):
    """Returns the shape of the target image."""
    return tuple((target_range[:, 1] - target_range[:, 0]).astype(int))


def _calc_target_range(source_shape, affine, round=True):
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


def _calc_target_range_p(target_shape, pivot, affine):
    """Calculates range of the target coordinates from pivot point and shape

    Args:
        target_shape (tuple): 3 :class:`int` spatial shape of the target image.
        pivot (tuple): 3 :class:`float` the center point in the source image.
            See :func:`reslice3e`.
        affine (numpy.ndarray): 4x4 affine matrix.

    Returns:
        numpy.ndarray: 3x2 matrix of the target coordinate ranges for the 3
            axes. The first column is the range starts and the second column is
            the range stops (the largest coordinates + 1).

    """
    target_shape = np.reshape(target_shape, (3, 1))
    pivot = np.reshape(pivot, (3, 1))
    pivot = convert_points_to_homogeneous(pivot)
    target_pivot = (affine @ pivot)[:3, :]
    starts = np.round(target_pivot - target_shape / 2.0)
    stops = starts + target_shape
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
    target_range = _calc_target_range(image_shape, affine, round=True)
    target_shape = _calc_target_shape(target_range)
    return target_shape
