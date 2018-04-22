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
        source_bbox ((3,) list of slice): The bbox in the source image
        target_bbox ((3,) list of slice): The bbox in the target image

    NOTE:
        Use uncrop3d with returned source_bbox and target_bbox to unresize

    """
    bbox = [slice(0, s) for s in image.shape]
    resized_bbox = resize_bbox3d(bbox, target_shape, allow_smaller=True)
    result, source_bbox, target_bbox = crop3d(image, resized_bbox)
    return result, source_bbox, target_bbox
