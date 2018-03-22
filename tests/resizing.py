#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing.resizing import padcrop3d


parser = argparse.ArgumentParser(description='Test resize')
parser.add_argument('image', help='The 3D image to resize')
parser.add_argument('target_shape', nargs=3, type=int, help='The target shape')
args = parser.parse_args()

image = nib.load(args.image).get_data()
resized = padcrop3d(image, args.target_shape)

shape = image.shape
slice_indices = np.array(shape) // 2
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image[slice_indices[0], :, :].T)
plt.subplot(2, 3, 2)
plt.imshow(image[:, slice_indices[1], :].T)
plt.subplot(2, 3, 3)
plt.imshow(image[:, :, slice_indices[2]].T)

shape = resized.shape
slice_indices = np.array(shape) // 2
plt.subplot(2, 3, 4)
plt.imshow(resized[slice_indices[0], :, :].T)
plt.subplot(2, 3, 5)
plt.imshow(resized[:, slice_indices[1], :].T)
plt.subplot(2, 3, 6)
plt.imshow(resized[:, :, slice_indices[2]].T)
plt.show()
