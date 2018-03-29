#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.cropping import crop3d, calc_bbox3d, resize_bbox3d
from image_processing_3d.cropping import uncrop3d


parser = argparse.ArgumentParser(description='Test cropping')
parser.add_argument('image', help='The 3D image to crop')
parser.add_argument('mask', help='The 3D mask to construct bbox from')
parser.add_argument('bbox_shape', nargs=3, type=int, help='The cropping bbox')
parser.add_argument('-s', '--allow-smaller', action='store_true', default=False,
                    required=False, help='Allow the bbox smaller than the mask')
parser.add_argument('-d', '--duplicated-channels', type=int, default=0,
                    required=False, help='Number of repeats to duplicate the '
                                         'image to multi-channels')
args = parser.parse_args()

image = nib.load(args.image).get_data()
if args.duplicated_channels > 0:
    image = np.repeat(image[None, ...], args.duplicated_channels, 0)
mask = nib.load(args.mask).get_data()
bbox = calc_bbox3d(mask)
resized_bbox = resize_bbox3d(bbox, args.bbox_shape, args.allow_smaller)
cropped, source_bbox, target_bbox = crop3d(image, resized_bbox)
uncropped = uncrop3d(cropped, image.shape, source_bbox, target_bbox)

if args.duplicated_channels == 0:
    image = image[None, ...]
    cropped = cropped[None, ...]
    uncropped = uncropped[None, ...]

for im, c, uc in zip(image, cropped, uncropped):

    plt.figure()

    shape = im.shape[-3:]
    slice_indices = np.array(shape) // 2
    plt.subplot(3, 3, 1)
    plt.imshow(im[slice_indices[0], :, :].T)
    plt.subplot(3, 3, 2)
    plt.imshow(im[:, 150, :].T)
    plt.subplot(3, 3, 3)
    plt.imshow(im[:, :, 40].T)

    shape = cropped.shape[-3:]
    slice_indices = np.array(shape) // 2
    plt.subplot(3, 3, 4)
    plt.imshow(c[slice_indices[0], :, :].T)
    plt.subplot(3, 3, 5)
    plt.imshow(c[:, slice_indices[1], :].T)
    plt.subplot(3, 3, 6)
    plt.imshow(c[:, :, slice_indices[2]].T)

    shape = image.shape[-3:]
    slice_indices = np.array(shape) // 2
    plt.subplot(3, 3, 7)
    plt.imshow(uc[slice_indices[0], :, :].T)
    plt.subplot(3, 3, 8)
    plt.imshow(uc[:, 150, :].T)
    plt.subplot(3, 3, 9)
    plt.imshow(uc[:, :, 40].T)

plt.show()
