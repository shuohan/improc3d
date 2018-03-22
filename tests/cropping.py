#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing.cropping import crop3d, calc_bbox3d


parser = argparse.ArgumentParser(description='Test cropping')
parser.add_argument('image', help='The 3D image to crop')
parser.add_argument('mask', help='The 3D mask to construct bbox from')
parser.add_argument('bbox_shape', nargs=3, type=int, help='The cropping bbox')
parser.add_argument('-s', '--allow-smaller', action='store_true', default=False,
                    required=False, help='Allow the bbox smaller than the mask')
args = parser.parse_args()

image = nib.load(args.image).get_data()
mask = nib.load(args.mask).get_data()
bbox = calc_bbox3d(mask, args.bbox_shape, args.allow_smaller)
cropped = crop3d(image, bbox)

plt.figure()

shape = image.shape
slice_indices = np.array(shape) // 2
plt.subplot(2, 3, 1)
plt.imshow(image[slice_indices[0], :, :].T)
plt.subplot(2, 3, 2)
plt.imshow(image[:, slice_indices[1], :].T)
plt.subplot(2, 3, 3)
plt.imshow(image[:, :, slice_indices[2]].T)

shape = cropped.shape
slice_indices = np.array(shape) // 2
plt.subplot(2, 3, 4)
plt.imshow(cropped[slice_indices[0], :, :].T)
plt.subplot(2, 3, 5)
plt.imshow(cropped[:, slice_indices[1], :].T)
plt.subplot(2, 3, 6)
plt.imshow(cropped[:, :, slice_indices[2]].T)

plt.show()
