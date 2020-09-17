#!/usr/bin/env python

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.crop import crop3d2, uncrop3d2


def test_crop():
    image_shape = (256, 135, 218)
    image = np.random.rand(*image_shape)
    bbox = (slice(-7, 250), slice(-11, 260), slice(10, 200))
    cropped, pad_width, cbbox = crop3d2(image, bbox)
    assert cropped.shape == (257, 271, 190)
    assert pad_width == ((7, 0), (11, 125), (0, 0))
    assert cbbox == (slice(0, 257), slice(0, 271), slice(10, 200))

    uncropped = uncrop3d2(cropped, image_shape, pad_width, cbbox)
    assert uncropped.shape == image_shape
    assert np.array_equal(uncropped[0 : 250, 0 : 135, 10 : 200],
                          image[0 : 250, 0 : 135, 10 : 200])

    image_mc = np.repeat(image[None, ...], 5, axis=0)
    cropped_mc, pad_width_mc, cbbox_mc = crop3d2(image_mc, bbox)
    uncropped_mc = uncrop3d2(cropped_mc, image_shape, pad_width_mc, cbbox_mc)
    assert pad_width_mc[1:] == pad_width
    assert cbbox_mc[1:] == cbbox
    assert pad_width_mc[0] == (0, 0)
    assert cbbox_mc[0] == ...
    for i in range(image_mc.shape[0]):
        assert np.array_equal(cropped_mc[i], cropped)
        assert np.array_equal(uncropped_mc[i], uncropped)

    print('Successful.')


if __name__ == '__main__':
    test_crop()
