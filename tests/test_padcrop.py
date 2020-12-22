#!/usr/bin/env python

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from improc3d import padcrop3d, uncrop3d


parser = argparse.ArgumentParser(description='Test resize')
parser.add_argument('image', help='The 3D image to resize')
parser.add_argument('target_shape', nargs=3, type=int, help='The target shape')
args = parser.parse_args()

image = nib.load(args.image).get_data()
resized, source_bbox, target_bbox = padcrop3d(image, args.target_shape)
unresized = uncrop3d(resized, image.shape, source_bbox, target_bbox)
diff = image - unresized

shape = image.shape
slice_indices = np.array(shape) // 2
plt.figure()
plt.subplot(3, 3, 1)
plt.imshow(image[slice_indices[0], :, :].T)
plt.subplot(3, 3, 2)
plt.imshow(image[:, slice_indices[1], :].T)
plt.subplot(3, 3, 3)
plt.imshow(image[:, :, slice_indices[2]].T)

shape = resized.shape
slice_indices = np.array(shape) // 2
plt.subplot(3, 3, 4)
plt.imshow(resized[slice_indices[0], :, :].T)
plt.subplot(3, 3, 5)
plt.imshow(resized[:, slice_indices[1], :].T)
plt.subplot(3, 3, 6)
plt.imshow(resized[:, :, slice_indices[2]].T)

shape = diff.shape
slice_indices = np.array(shape) // 2
plt.subplot(3, 3, 7)
plt.imshow(diff[slice_indices[0], :, :].T)
plt.subplot(3, 3, 8)
plt.imshow(diff[:, slice_indices[1], :].T)
plt.subplot(3, 3, 9)
plt.imshow(diff[:, :, slice_indices[2]].T)

plt.show()
