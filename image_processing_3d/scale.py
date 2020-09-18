import numpy as np
from scipy.ndimage.interpolation import map_coordinates

from .reslice import reslice3d


def scale3d(image, x_scale, y_scale, z_scale, pivot=None, order=1,
            use_source_shape=True):
    """Scales a 3D image around a point.

    This function scales a 3D image around a point. If the input image is a
    channel-first multi-channel 3D image (i.e. 4D image with the first dimension
    as channels), this funcition uses the same factors to scale the image for
    each channel.
    
    Args:
        image (numpy.ndarray): The 3D or 4D image to scale. Channel first if 4D.
        x_scale (float): The scaling factor along x axis.
        y_scale (float): The scaling factor along y axis.
        z_scale (float): The scaling factor along z axis.
        pivot (iterable, optional): The 3D scaling center point. If ``None``,
            use the image center as the scaling center. Otherwise, it can be a
            :py:class:`tuple` or :class:`numpy.ndarray` of :class:float.
        order (int, optional): The interpolation order.
        use_source_shape (bool, optional): Use the source image shape as the
            transformed image shape if ``True``.

    Returns:
        numpy.ndarray: The 3D or 4D scaled image.

    """
    scaling = np.array([[x_scale, 0, 0, 0],
                        [0, y_scale, 0, 0],
                        [0, 0, z_scale, 0],
                        [0, 0, 0, 1]])
    target_shape = image.shape if use_source_shape else None
    return reslice3d(image, scaling, order, target_shape, pivot)
