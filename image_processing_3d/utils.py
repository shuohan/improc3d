import numpy as np
from scipy.ndimage.interpolation import map_coordinates


def convert_transformation_to_homogeneous(trans):
    """Converts the 3x3 transformation matrix into the homogeneous coordinate.

    Args:
        trans (numpy.ndarray): The 3D transformation matrix to convert.
    
    Returns:
        numpy.ndarray: Homogeneous rotation matrix.

    """
    result = np.eye(4)
    result[:3, :3] = trans
    return result


def convert_translation_to_homogeneous(translation):
    """Converts 3D translation into the homogeneous coordinate.

    Args:
        translation (numpy.ndarray): The 3D translation to convert.

    Returns:
        numpy.ndarray: Translation in the homogeneous coordinate.

    """
    result = np.eye(4)
    result[:3, 3] = translation
    return result


def convert_points_to_homogeneous(points):
    """Converts 3D points into the homogeneous coordinate

    Args:
        points (numpy.ndarray): The points to convert. It should be a 2D array
            with shape 3 x num_points.

    Returns:
        numpy.ndarray: Points in the homogeneous coordinate. A 2D array with
        shape 4 x num_points.

    """
    points = np.vstack([points, np.ones((1, points.shape[1]))])
    return points


def convert_points_from_homogeneous(points):
    """Converts 3D points from the homogeneous coordinate

    Args:
        points (numpy.ndarray): The points to convert. It should a 2D array with
            shape 4 x num_points.

    Returns:
        numpy.ndarray: Non-homogeneous points. A 2D array with shape
        3 x num_points.

    """
    return points[:3, :]


def convert_grid_to_coords(grid):
    """Converts a meshgrid to coordinate points.

    This function converts the meshgrid which is the indices of the image pixels
    along x, y, and z axes, to a matrix which is stacked coordinate vectors.

    For example, it converts::
    
        [[1, 2],    [[5, 6],
         [3, 4]]     [7, 8]]

    to::

        [[1, 2, 3, 4]
         [5, 6, 7, 8]]
    
    Args:
        grid (tuple): The meshgrid of image voxel indices. The number of element
            is equal to the dimention of the image and each element is the
            :class:`numpy.ndarray` coordinates along an axis.

    Returns:
        numpy.ndarray: The num_dims x num_pixels coordinate vectors.

    """
    coords = np.vstack([g.flatten()[None, ...] for g in grid]) 
    return coords


def calc_image_coords(shape, homogeneous=True):
    """Calculates the coordinates of all image voxels.

    Args:
        shape (tuple or numpy.ndarray): The 3-element :py:class:`int` spatial
            shape of the image, or 3 x 2 array whose first column is the
            coordinate starts and the second column is the stops (the largest
            coordinates + 1).
        homogeneous (bool, optional): Convert the coordinates into the
            homogeneous coordinate if ``True``.

    Returns:
        numpy.ndarray: The num_dims x num_pixels coordinate vectors.

    Raises:
        RuntimeError: Invalid input ``shape``.

    """
    shape = np.array(shape)
    if len(shape.shape) == 1:
        g = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    elif len(shape.shape) == 2 and shape.shape[1] == 2:
        g = np.meshgrid(*[np.arange(a, b) for (a, b) in shape], indexing='ij')
    else:
        raise RuntimeError('Invalid shape %s' % shape.__repr__())

    coords = convert_grid_to_coords(g)
    if homogeneous == True:
        coords = convert_points_to_homogeneous(coords)
    return coords


def interp_image(image, source_coords, order=1):
    """Interpolates a 3D image with coordinates.

    This function assumes 0 outside the image. If the input image is 4D, it
    assumes channel-first and interpolates the image at the same coordinates for
    each channel.

    Args:
        image (numpy.ndarray): The image to interpolate; channel first if 4D.
        source_coords (numpy.ndarray): 3 x num_points coordinates to interpolate
            from the image. Each column is a 3D point.
        order (int, optional): The interpolation order. See
            :func:`scipy.ndimage.interpolation.map_coordinates`.

    Returns:
        numpy.ndarray: An 1D array of the values at each source coordinate.

    """
    if len(image.shape) == 4:
        result = [map_coordinates(m, source_coords, order=order) for m in image]
        result = np.vstack(result)
    else:
        result = map_coordinates(image, source_coords, order=order)
    return result
