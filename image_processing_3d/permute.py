"""Functions and classes to permute an image.

"""
import numpy as np
from enum import IntEnum


class Axis(IntEnum):
    """The axis index of a 3D image.

    Attributes:
        X (int): The first axis.
        Y (int): The second axis.
        Z (int): The third axis.

    """
    X = 0
    Y = 1
    Z = 2


def permute3d(image, x=Axis.X, y=Axis.Y, z=Axis.Z, return_inv_axes=True):
    """Permutes a 3D image.

    Example:
        To permute back to the original axis order:

        >>> perm, inv_x, inv_y, inv_z = permute3d(image, x, y, z)
        >>> image2 = permute3d(perm, inv_x, inv_y, inv_z, return_inv_axes=False)
        >>> assert np.array_equal(image, image2)

    Args:
        image (numpy.ndarray): The image to permute.
        x (Axis or int): The axis in the input ``image`` to permute to the
            x-axis in the result image.
        y (Axis or int): The axis in the input ``image`` to permute to the
            y-axis in the result image.
        z (Axis or int): The axis in the input ``image`` to permute to the
            z-axis in the result image.
        return_inv_axes (bool): If ``True``, return the inverse axes to permute
            the data back.

    Returns
    -------
    perm_image: numpy.ndarray
        The permuted image.
    inv_x: Axis
        The axis in the permuted image corresponding to the x-axis in the
        original image.
    inv_y: Axis
        The axis in the permuted image corresponding to the y-axis in the
        original image.
    inv_z: Axis
        The axis in the permuted image corresponding to the z-axis in the
        original image.

    Raises:
        RuntimeError: The input ``x``, ``y``, and ``z`` contain the same values.

    """
    x = Axis(x) if type(x) is int else x
    y = Axis(y) if type(y) is int else y
    z = Axis(z) if type(z) is int else z
    if len({x, y, z}) != 3:
        raise RuntimeError('Inputs "x", "y", and "z" have the same value.')
    perm_axes = [axis.value for axis in (x, y, z)]
    perm_image = np.transpose(image, perm_axes)

    if return_inv_axes:
        perm_mat = _calc_perm_mat_from_axes(x, y, z)
        inv_perm_mat = np.linalg.inv(perm_mat)
        inv_x = _get_axis_from_perm_mat(0, inv_perm_mat)
        inv_y = _get_axis_from_perm_mat(1, inv_perm_mat)
        inv_z = _get_axis_from_perm_mat(2, inv_perm_mat)
        return perm_image, inv_x, inv_y, inv_z
    else:
        return perm_image


def _calc_perm_mat_from_axes(x, y, z):
    """Calculates the permutation matrix from the axes in the target image."""
    perm_mat = np.zeros((3, 3), dtype=int)
    perm_mat[0, x.value] = 1
    perm_mat[1, y.value] = 1
    perm_mat[2, z.value] = 1
    return perm_mat


def _get_axis_from_perm_mat(targ_axis, perm_mat):
    """Returns the corresponding axis in the original image."""
    return Axis(np.argmax(perm_mat[targ_axis, ...]))
