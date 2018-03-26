# -*- coding: utf-8 -*-

import numpy as np

from .cropping import crop3d, resize_bbox3d


def padcrop3d(image, target_shape):
    """Pad or crop the 3D image to resize

    Args:
        image (3D numpy.array): The image to pad
        target_shape ((3,) tuple of int): The shape of the resized image

    Returns:
        result (3D numpy.array): The resized image

    """
    bbox = [slice(0, s) for s in image.shape]
    resized_bbox = resize_bbox3d(bbox, target_shape, allow_smaller=True)
    result = crop3d(image, resized_bbox)[0]
    return result
