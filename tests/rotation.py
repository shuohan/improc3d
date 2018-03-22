#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing.rotation import rotate3d


parser = argparse.ArgumentParser(description='Test rotation')
parser.add_argument('image', help='The 3D image to rotate')
parser.add_argument('-x', '--x-angle', default=5, required=False, type=float,
                    help='The rotation angle around x axis in degree')
parser.add_argument('-y', '--y-angle', default=5, required=False, type=float,
                    help='The rotation angle around y axis in degree')
parser.add_argument('-z', '--z-angle', default=5, required=False, type=float,
                    help='The rotation angle around z axis in degree')
parser.add_argument('-p', '--point', default=None, required=False, nargs=3,
                    type=float, help='The 3D image to rotate')
args = parser.parse_args()

image = nib.load(args.image).get_data()
point = np.array(args.point)
rotated = rotate3d(image, args.x_angle, args.y_angle, args.z_angle, point)
shape = image.shape
slice_indices = np.array(shape) // 2
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image[slice_indices[0], :, :].T)
plt.subplot(2, 3, 2)
plt.imshow(image[:, slice_indices[1], :].T)
plt.subplot(2, 3, 3)
plt.imshow(image[:, :, slice_indices[2]].T)
plt.subplot(2, 3, 4)
plt.imshow(rotated[slice_indices[0], :, :].T)
plt.subplot(2, 3, 5)
plt.imshow(rotated[:, slice_indices[1], :].T)
plt.subplot(2, 3, 6)
plt.imshow(rotated[:, :, slice_indices[2]].T)
plt.show()
