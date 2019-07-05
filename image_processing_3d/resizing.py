# -*- coding: utf-8 -*-

import numpy as np

from .cropping import crop3d, resize_bbox3d


def padcrop3d(image, target_shape):
    """Pads or crops the 3D image to resize.

    This function pads zero to the image if ``target_shape`` exceeds ``image``
    along an axis and crop it if ``target_shape`` is contained within the
    ``image``.

    NOTE:
        Use :func:`image_processing_3d.uncrop3d` with the returned
        ``source_bbox`` and ``target_bbox`` by this function  to unresize.

    Args:
        image (numpy.ndarray): The 3D or 4D image to pad or crop; if 4D, the
            first dimension is assumed to be channels.
        target_shape (tuple): The 3 :class:`int` shape of the resized image.

    Returns
    -------
    result: numpy.ndarray
        The 3D or 4D resized image.
    source_bbox: tuple
        The 3 :class:`slice` bounding box in the source image.
    target_bbox: tuple
        The 3 :class:`slice` bounding box in the target image.

    """
    if len(image.shape) == 3:
        shape = image.shape
    elif len(image.shape) == 4:
        shape = image.shape[1:]
    bbox = [slice(0, s) for s in shape]
    resized_bbox = resize_bbox3d(bbox, target_shape, allow_smaller=True)
    result, source_bbox, target_bbox = crop3d(image, resized_bbox)
    return result, source_bbox, target_bbox
