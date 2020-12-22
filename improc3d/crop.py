import numpy as np
from scipy.ndimage.measurements import find_objects


def crop3d(image, bbox, pad='zero', return_bbox=True):
    """Crops a 3D image using a bounding box.

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
        image (numpy.ndarray): The 3D or 4D image to crop. If ``image`` is 4D,
            the 0 dimension is assumed to be the channels and the same bounding
            box will be applied to all the channels.
        bbox (tuple[slice]): The length==3 bounding box. The start and stop of
            each slice should NOT be ``None``.
        return_bbox (bool): Output ``source_bbox`` and ``target_bbox`` if true.

    Returns
    -------
    cropped: numpy.ndarray
        The 3D or 4D cropped image. If ``image`` is 4D, ``cropped`` is also 4D
        and channel first.
    source_bbox: tuple
        The 3 :class:`slice` bounding box in the source image.
    target_bbox: tuple
        The 3 :class:`slice` bounding box in the target image.

    """
    num_dims = len(bbox)
    source_shape = image.shape[-num_dims:]
    target_shape = [b.stop - b.start for b in bbox]
    source_bbox = _calc_source_bounding_box(bbox, source_shape)
    target_bbox = _calc_target_bounding_box(bbox, source_shape, target_shape)
    if len(image.shape) == 4:
        target_shape = [image.shape[0]] + target_shape
        source_bbox = tuple([...] + list(source_bbox))
        target_bbox = tuple([...] + list(target_bbox))
    if pad == 'zero':
        cropped = np.zeros(target_shape, dtype=image.dtype)
    elif pad == 'orig':
        cropped = np.ones(target_shape, dtype=image.dtype) * image.flatten()[0]
    cropped[tuple(target_bbox)] = image[tuple(source_bbox)]

    if return_bbox:
        return cropped, source_bbox, target_bbox
    else:
        return cropped


def _calc_source_bounding_box(bbox, source_shape):
    """Calculates the bounding of the source image to crop.

    The data of the image within this source bounding is extracted for
    cropping.

    Args:
        bbox (tuple[slice]): The len==3 bounding box of the cropping. The start
            of the slice could be negative meaning to pad zeros on the left; the
            stop of the slice could be greater than the size of the image along
            this direction, which means to pad zeros on the right.
        source_shape (tuple[int]): The spatial shape of the image to crop.

    Returns:
        tuple[slice]: The bounding box used to extract data from the source
            image to crop.

    """
    source_bbox = list()
    for bounding, source_size in zip(bbox, source_shape):
        source_start = max(bounding.start, 0)
        source_stop = min(bounding.stop, source_size)
        source_bbox.append(slice(source_start, source_stop, None))
    return tuple(source_bbox)


def _calc_target_bounding_box(bbox, source_shape, target_shape):
    """Calculates the bounding of the cropped target image.

    ``bbox`` is relative to the shape of the source image. For the target
    image, the number of pixels on the left is equal to the absolute value of
    the negative start (if any), and the number of pixels on the right is equal
    to the number of pixels target size exceeding the source size.

    Args:
        bbox (tuple[slice]): The len==3 bounding box for the cropping. The start
            of the slice can be negative, meaning to pad zeros on the left; the
            stop of the slice can be greater than the size of the image along
            this direction, meaning to pad zeros on the right.
        source_shape (tuple[int]): The spatial shape of the image to crop.
        target_shape (tuple[int]): The spatial shape of the cropped image.

    Returns:
        tuple[slice]: The bounding box of the cropped image used to put the
            extracted data from the source image into the traget image.

    """
    target_bbox = list()
    for bounding, ssize, tsize in zip(bbox, source_shape, target_shape):
        target_start = 0 - min(bounding.start, 0)
        target_stop = tsize - max(bounding.stop - ssize, 0)
        target_bbox.append(slice(target_start, target_stop, None))
    return tuple(target_bbox)


def calc_bbox3d(mask):
    """Calculates the bounding box surrounding the mask.

    This function calcualtes the bounding boxes of all connected components in
    the mask then unites all of them together to get a single bounding box.

    Args:
        mask (numpy.ndarray): The 3D or 4D mask to calculate the bounding
            box from; if 4D, the first dimension is assumed to be channels and
            only the first channel is used to calculate the bounding box since
            the same mask should be applied to all channels of an image.

    Returns:
        tuple[slice]: The length==3 bounding box around the mask.

    """
    mask = mask.astype(bool)
    if len(mask.shape) == 4:
        mask = mask[0, ...]
    bboxes = find_objects(mask)
    starts = [[s.start for s in bbox] for bbox in bboxes]
    stops = [[s.stop for s in bbox] for bbox in bboxes]
    starts = np.min(starts, axis=0)
    stops = np.max(stops, axis=0)
    bbox = [slice(start, stop, None) for start, stop in zip(starts, stops)]
    return tuple(bbox)


def resize_bbox3d(bbox, bbox_shape, allow_smaller=True):
    """Resizes a bounding box to a certain shape.

    If the ``bbox_shape`` is larger than ``bbox``, the left and right of
    ``bbox`` is padded by the same amount of extra space. If ``bbox_shape``
    is smaller than ``bbox``, the left and right of ``bbox`` is cropped.

    Args:
        bbox (tuple[slice]): The length==3 bounding box to resize.
        bbox_shape (tuple[int]): The length==3 spatial shape of the resized bbox.
        allow_smaller (bool): Allow ``bbox_shape`` is smaller than ``bbox``.

    Returns:
        tuple[int]: The length==3 resized bounding box. If multi-channel, the
            output has 4 element and the first (0th) is ``...``.

    Raises:
        RuntimeError: ``bbox_shape`` is smaller than ``bbox`` if
            ``allow_smaller`` is ``False``.

    """
    resized_bbox = list()
    for source_bound, target_size in zip(bbox, bbox_shape):
        source_size = source_bound.stop - source_bound.start
        diff = target_size - source_size
        if diff < 0 and not allow_smaller:
            raise RuntimeError('Target shape should be bigger than the '
                               'source shape')
        else:
            left_padding = diff // 2
            right_padding = diff - left_padding
            target_bound = slice(source_bound.start - left_padding,
                                 source_bound.stop + right_padding)
        resized_bbox.append(target_bound)
    return tuple(resized_bbox)


def uncrop3d(image, source_shape, source_bbox, target_bbox):
    """Reverses :func:`crop3d` but pads zeros around the cropped region.

    Note:
        ``source_bbox`` and ``target_bbox`` should be from the outputs of
        :func:`crop3d`.

    Args:
        image (numpy.ndarray): The 3D or 4D image to uncrop; channels first if
            ``image`` is 4D.
        source_shape (tuple[int]): The len==3 spatial shape of uncropped image.
        source_bbox (tuple[slice]): The length==3 bounding box used to crop the
            original image.
        target_bbox (tuple[slice]): The length==3 corresponding bounding box in
            the cropped image.

    Returns:
        numpy.ndarray: The 3D or 4D uncropped image; channels first if 4D.

    """
    uncropped = np.zeros(source_shape, dtype=image.dtype)
    uncropped[tuple(source_bbox)] = image[tuple(target_bbox)]
    return uncropped


def padcrop3d(image, target_shape, return_bbox=True):
    """Pads or crops the 3D image to resize.

    This function pads zero to the image if ``target_shape`` exceeds ``image``
    along an axis and crops it if ``target_shape`` is contained within the
    ``image``.

    NOTE:
        Use :func:`uncrop3d` with the returned ``source_bbox`` and
        ``target_bbox`` by this function to unresize.

    Args:
        image (numpy.ndarray): The 3D or 4D image to pad or crop; if 4D, the
            first dimension is assumed to be channels.
        target_shape (tuple[int]): The length==3 spatial shape of the resized
            image.
        return_bbox (bool): Output ``source_bbox`` and ``target_bbox`` if true.

    Returns
    -------
    result: numpy.ndarray
        The 3D or 4D resized image.
    source_bbox: tuple[slice]
        The length==3 bounding box in the source image.
    target_bbox: tuple[slice]
        The length==3 bounding box in the target image.

    """
    bbox = _calc_padcrop_bbox(image, target_shape)
    return crop3d(image, bbox, return_bbox=return_bbox)


def _calc_padcrop_bbox(image, target_shape):
    """Calculates the bbox for :func:`padcrop3d`."""
    shape = image.shape if len(image.shape) == 3 else image.shape[1:]
    bbox = [slice(0, s) for s in shape]
    resized_bbox = resize_bbox3d(bbox, target_shape, allow_smaller=True)
    return resized_bbox


def crop3d2(image, bbox, mode='constant', return_bbox=True, **kwargs):
    """Crops a 3D image first with numpy.pad then crop.

    Args:
        image (numpy.ndarray): The 3D or 4D image to crop. If ``image`` is 4D,
            the 0 dimension is assumed to be the channels and the same bounding
            box will be applied to all the channels.
        bbox (tuple[slice]): The length==3 bounding box specifying the cropping
            range. The start and stop of each slice should not be ``None``.
        mode (str): The padding mode. See :func:`numpy.pad` for more details.
        return_bbox (bool): Output ``pad_width`` and ``cropping_bbox`` if true.
        kwargs (dict): The other parameters of :func:`numpy.pad`.

    Returns
    -------
    cropped_image: numpy.ndarray
        The 3D or 4D cropped image. If ``image`` is 4D, ``cropped_image`` is
        also 4D and channel first.
    pad_width: tuple[tuple[int]]
        The paddings use to pad the input image.
    cropping_bbox: tuple[slice]
        The bounding box used to crop the padded image.

    """
    source_shape = image.shape[-len(bbox):]
    left_pads = [max(0, 0 - b.start) for b in bbox]
    right_pads = [max(0, b.stop - s) for s, b in zip(source_shape, bbox)]
    pad_width = list(zip(left_pads, right_pads))
    cropping_bbox = [slice(b.start + l, b.stop + l)
                     for b, l, r in zip(bbox, left_pads, right_pads)]
    if image.ndim == 4:
        pad_width.insert(0, (0, 0))
        cropping_bbox.insert(0, ...)
    cropping_bbox = tuple(cropping_bbox)
    pad_width = tuple(pad_width)
    padded_image = np.pad(image, pad_width, mode=mode, **kwargs)
    cropped_image = padded_image[cropping_bbox]

    if return_bbox:
        return cropped_image, pad_width, cropping_bbox
    else:
        return cropped_image


def uncrop3d2(image, source_shape, pad_width, cropping_bbox):
    """Reverses :func:`crop3d2` but pads zeros around the cropped region.

    Note:
        The ``pad_width`` and ``cropping_bbox`` should be the outputs of the
        function :func:`crop3d2`.

    Args:
        image (numpy.ndarray): The 3D or 4D image to uncrop; channels first if
            ``image`` is 4D.
        source_shape (tuple[int]): The len==3 spatial shape of uncropped image.
        pad_width (tuple[tuple[int]]): The paddings used to pad the input image.
        cropping_bbox (tuple[slice]): The bbox used to crop the padded image.

    Returns:
        numpy.ndarray: The 3D or 4D uncropped image; channels first if 4D.

    """
    source_shape = list(source_shape)
    if image.ndim == 4:
        source_shape.insert(0, image.shape[0])
    padded_shape = [ss + l + r for ss, (l, r) in zip(source_shape, pad_width)]
    padded_image = np.zeros(padded_shape, dtype=image.dtype)
    padded_image[cropping_bbox] = image
    bbox = [slice(l, s - r) for (l, r), s in zip(pad_width, padded_image.shape)]
    uncropped_image = padded_image[tuple(bbox)]
    return uncropped_image



def padcrop3d2(image, target_shape, mode='constant', return_bbox=True,
               **kwargs):
    """Pads the image with :func:`numpy.pad` then crops the 3D image to resize.

    This function pads values to the image if ``target_shape`` exceeds ``image``
    along an axis and crops it if ``target_shape`` is contained within the
    ``image``.

    NOTE:
        Use :func:`uncrop3d2` with the returned ``pad_width`` and
        ``cropping_bbox`` by this function to unresize.

    Args:
        image (numpy.ndarray): The 3D or 4D image to pad or crop; if 4D, the
            first dimension is assumed to be channels.
        target_shape (tuple[int]): The length==3 spatial shape of the resized
            image.
        mode (str): The padding mode. See :func:`numpy.pad` for more details.
        return_bbox (bool): Output ``pad_width`` and ``cropping_bbox`` if true.
        kwargs (dict): The other parameters of :func:`numpy.pad`.

    Returns
    -------
    result: numpy.ndarray
        The 3D or 4D resized image.
    pad_width: tuple[tuple[int]]
        The paddings use to pad the input image.
    cropping_bbox: tuple[slice]
        The bounding box used to crop the padded image.

    """
    bbox = _calc_padcrop_bbox(image, target_shape)
    return crop3d2(image, bbox, mode=mode, return_bbox=return_bbox, **kwargs)
