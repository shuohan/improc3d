import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .utils import convert_grid_to_coords, interp_image


def deform3d(image, x_deformation, y_deformation, z_deformation, order=1):
    """Transforms a 3D image using a deformation field.

    This function assumes 0 for the voxel values outside the image. If the input
    image is 4D, it assuems channel-first and applies the same deformation to
    each of the channels.
    
    Args:
        image (numpy.ndarray): The 3D or 4D image to deform. Channel first if it
            is 4D.
        x_deformation (numpy.ndarray): The 3D x deformation. Per-voxel
            translation along the x axis.
        y_deformation (numpy.ndarray): The 3D y deformation. Per-voxel
            translation along the y axis.
        z_deformation (numpy.ndarray): The 3D z deformation. Per-voxel
            translation along the z axis.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.

    Returns:
        numpy.ndarray: The 3D or 4D deformed image.

    """
    target_grid = np.meshgrid(*[np.arange(s) for s in image.shape[-3:]],
                              indexing='ij')
    deformation = [x_deformation, y_deformation, z_deformation]
    source_grid = [g - d for g, d in zip(target_grid, deformation)]
    source_coords = convert_grid_to_coords(source_grid)
    result = interp_image(image, source_coords, order=order)
    return np.reshape(result, image.shape)


def calc_random_deformation3d(image_shape, sigma, scale):
    """Calculates a component of a random deformation field.

    This deformation is along one axis. Call this function three times from
    deformation along x, y, and z axes. Check the source code for details of the
    computation.

    Args:
        image_shape (tuple): The 3 :class:`int` spatial shape of the image.
        sigma (float): The value controling the smoothness of the deformation
            field. Larger the value is, smoother the field.
        scale (float): The deformation is supposed to draw from a uniform
            distribution [-eps, +eps]. Use this value to specify the upper bound
            of the sampling distribution. Larger the value is, stronger the
            deformation.

    Returns:
        numpy.ndarray: One component of the 3D deformation filed along an axis.

    """
    result = np.random.rand(*image_shape) * 2 - 1
    result = gaussian_filter(result, sigma)
    result = result / np.max(result) * scale
    return result
