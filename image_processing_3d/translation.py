# -*- coding: utf-8 -*-

import numpy as np


def translate3d_int(image, x_trans, y_trans, z_trans):
    """Translate 3d image with integer translation
    
    Outside the image is assumed to be 0.

    Args:
        image (3D/4D numpy.array): The 3D image to translate. If not 3D, the
            first axis is assuemd to be channels like RGB
        x_trans (int/float): The translation along x axis. If not integer, the
            translation is rounded 
        y_trans (int/float): The translation along y axis. If not integer, the
            translation is rounded 
        z_trans (int/float): The translation along z axis. If not integer, the
            translation is rounded 

    Returns:
        translated (3D/4D numpy.array): The translated image

    """
    x_trans = int(np.round(x_trans))
    y_trans = int(np.round(y_trans))
    z_trans = int(np.round(z_trans))

    xsize, ysize, zsize = image.shape[-3:]
    x_source_slice, x_target_slice = _calc_index(x_trans, xsize)
    y_source_slice, y_target_slice = _calc_index(y_trans, ysize)
    z_source_slice, z_target_slice = _calc_index(z_trans, zsize)

    translated = np.zeros(image.shape, dtype=image.dtype)
    translated[..., x_target_slice, y_target_slice, z_target_slice] = \
            image[..., x_source_slice, y_source_slice, z_source_slice]

    return translated


def _calc_index(trans, size):
    """Calculate target and source indexing slices from translation

    Args:
        trans (int): The translation of the data
        size (int): The size of the data

    Returns:
        source (slice): The indexing slice in the source data
        target (slice): The indexing slice in the target data

    """
    if trans > 0:
        source = slice(0, size-1-trans, None)
        target = slice(trans, size-1, None)
    elif trans <= 0:
        source = slice(-trans, size-1, None)
        target = slice(0, size-1+trans, None)
    return source, target
