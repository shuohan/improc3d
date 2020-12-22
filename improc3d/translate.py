import numpy as np


def translate3d_int(image, x_trans, y_trans, z_trans):
    """Translates a 3D image with integer translation.
    
    This function uses 0 as the voxel values outside the image. If the image is
    4D, it assumes the image is channel first and applies the same translation
    to each channel.

    Args:
        image (numpy.ndarray): The image to translate. Channel first if 4D.
        x_trans (int or float): The translation along x axis. If not
            :class:`int`, the translation is rounded.
        y_trans (int or float): The translation along y axis. If not 
            :class:`int`, the translation is rounded.
        z_trans (int or float): The translation along z axis. If not
            :class:`int`, the translation is rounded.

    Returns:
        numpy.ndarray: The 3D or 4D translated image.

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
    """Calculates target and source indexing :class:`slice` along a single axis.

    Args:
        trans (int): The translation of the data.
        size (int): The size of the data.

    Returns
    -------
    source: slice
        The indexing in the source data.
    target: slice
        The indexing in the target data.

    """
    if trans > 0:
        source = slice(0, size-1-trans, None)
        target = slice(trans, size-1, None)
    elif trans <= 0:
        source = slice(-trans, size-1, None)
        target = slice(0, size-1+trans, None)
    return source, target
