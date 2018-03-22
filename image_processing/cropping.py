# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage.measurements import find_objects


def crop3d(image, bbox):
    """Crop 3D image using a bounding box

    The size of bbox can be larger than the image. In that case, 0 will be put
    into the extra area. To copy the data within the bbox from the source image
    (the image to crop) to the target image (the cropped image), the cropped
    area with respect to the source image and the corresponding area in the
    target image can have different starts and stops although they have the same
    shape.

    For example, assume x is a 3-by-5 image and the bbox is (-1:3, 1:3) yielding
    a cropped image with size (5, 3). The cropping area with respect to the
    source image is then (0:2, 1:3) (extra area will be filled with 0) while the
    corresponing cropping area in the target image is (1:3, 0:2).

    Args:
        image (3D numpy.array): The image to crop. If `image` is 4D, the 0
            dimension is assumed to be the channels
        bbox ((3,) tuple of slice): The bounding box. `start` and `stop` of
            `slice` should not be `None`

    Returns:
        cropped (3D numpy.array): The cropped image. If `image` is 4D, `cropped`
            is also 4D and channel first.
        source_bbox ((3,) list of slice): The bbox in the source image
        target_bbox ((3,) list of slice): The bbox in the target image

    """
    num_dims = len(bbox)
    source_shape = image.shape[-num_dims:]
    target_shape = [b.stop - b.start for b in bbox]
    source_bbox = _calc_source_bounding_box(bbox, source_shape)
    target_bbox = _calc_target_bounding_box(bbox, source_shape, target_shape)
    if len(image.shape) == 4:
        target_shape = [image.shape[0]] + target_shape
        source_bbox = [...] + source_bbox
        target_bbox = [...] + target_bbox
    cropped = np.zeros(target_shape, dtype=image.dtype)
    cropped[target_bbox] = image[source_bbox]
    return cropped, source_bbox, target_bbox


def _calc_source_bounding_box(bbox, source_shape):
    """Calculate the bounding of the source image to crop

    The data of the image within this source bounding is extracted for
    cropping.

    Args:
        bbox ((3,) list of slice): The bounding box of the cropping. The
            start of the slice could be negative meaning to pad zeros on the
            left; the stop of the slice could be greater than the size of
            the image along this direction, which means to pad zeros on the
            right
        source_shape ((3,) tuple): The shape of the image to crop

    Returns:
        source_bbox ((3,) list of slice): The bounding box used to extract
            data from the source image to crop

    """
    source_bbox = list()
    for bounding, source_size in zip(bbox, source_shape):
        source_start = max(bounding.start, 0)
        source_stop = min(bounding.stop, source_size)
        source_bbox.append(slice(source_start, source_stop, None))
    return source_bbox


def _calc_target_bounding_box(bbox, source_shape, target_shape):
    """Calculate the bounding of the cropped target image

    `bbox` is relative to the shape of the source image. For the target
    image, the number of pixels on the left is equal to the absolute value of
    the negative start (if any), and the number of pixels on the right is equal
    to the number of pixels target size exceeding the source size.

    Args:
        bbox ((3,) list of slice): The bounding box of the cropping. The
            start of the slice could be negative meaning to pad zeros on the
            left; the stop of the slice could be greater than the size of
            the image along this direction, which means to pad zeros on the
            right
        source_shape ((3,) tuple): The shape of the image to crop
        target_shape ((3,) tuple): The shape of the cropped image

    Returns:
        target_bbox ((3,) list of slice): The bounding box of the cropped
            image used to put the extracted data from the source image into
            the traget image

    """
    target_bbox = list()
    for bounding, ssize, tsize in zip(bbox, source_shape, target_shape):
        target_start = 0 - min(bounding.start, 0)
        target_stop = tsize - max(bounding.stop - ssize, 0)
        target_bbox.append(slice(target_start, target_stop, None))
    return target_bbox


def calc_bbox3d(mask):
    """Calculate bounding box surrounding the mask

    Calcualte the bounding boxes of connected components in the mask and unite
    them all.

    Args:
        mask (3D numpy.array): The mask to calculate bbox from

    Returns:
        bbox (1x3 list of slice): Calculated bounding box

    """
    bboxes = find_objects(mask.astype(bool))
    starts = [[s.start for s in bbox] for bbox in bboxes]
    stops = [[s.stop for s in bbox] for bbox in bboxes]
    starts = np.min(starts, axis=0)
    stops = np.min(stops, axis=0)
    bbox = [slice(start, stop, None) for start, stop in zip(starts, stops)]
    return bbox


def resize_bbox3d(bbox, bbox_shape, allow_smaller=True):
    """Resize bbox to have bbox_shape

    If the `bbox_shape` is larger than `bbox`, the left and right of `bbox` is
    padded by the same amount of space. If the `bbox_shape` is smaller than
    `bbox`, the left and right of `bbox` is cropped by the same amount.

    Args:
        bbox ((3,) tuple of slice): The bbox to resize
        bbox_shape ((3,) tuple of int): The shape of the resized bbox
        allow_smaller (bool): Allow `bbox_shape` is smaller than `bbox`

    Returns:
        resized_bbox (1x3 list of slice): Resized bounding box

    Raises:
        RuntimeError: `bbox_shape` is smaller than the mask
    
    """
    resized_bbox = list()
    for source_bound, target_size in zip(bbox, bbox_shape):
        source_size = source_bound.stop - source_bound.start
        diff = target_size - source_size
        if diff < 0 and not allow_smaller:
            raise RuntimeError('Target shape should be bigger than the '
                               'source shape')
        else:
            left_padding = np.floor(diff / 2).astype(int)
            right_padding = np.ceil(diff / 2).astype(int)
            target_bound = slice(source_bound.start - left_padding,
                                 source_bound.stop + right_padding)
        resized_bbox.append(target_bound)
    return resized_bbox


def uncrop3d(image, source_shape, source_bbox, target_bbox):
    """Reverse crop3d but pad zeros around the cropped region
    
    Args:
        image (3D numpy.array): The image to uncrop
        source_shape ((3,) tuple): The shape of uncropped image
        source_bbox ((3,) list of slice): The bbox used to crop the image
        target_bbox ((3,) list of slice): The corresponding bbox in the cropped

    Returns:
        uncropped (3D numpy.array): Uncropped image

    """
    uncropped = np.zeros(source_shape, dtype=image.dtype)
    uncropped[source_bbox] = image[target_bbox]
    return uncropped
