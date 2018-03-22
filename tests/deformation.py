#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing.deformation import deform, calc_random_defromation


parser = argparse.ArgumentParser(description='Test random free deformation')
parser.add_argument('image', help='The 3D image to deform')
parser.add_argument('-s', '--sigma', default=3, required=False, type=float,
                    help='Control the smoothness of the deformation field')
parser.add_argument('-l', '--limit', default=3, required=False, type=float,
                    help='The maximum possible deformation in pixel')
args = parser.parse_args()

image = nib.load(args.image).get_data()

x_deformation = calc_random_defromation(image.shape, args.sigma, args.limit)
y_deformation = calc_random_defromation(image.shape, args.sigma, args.limit)
z_deformation = calc_random_defromation(image.shape, args.sigma, args.limit)
deformed = deform(image, x_deformation, y_deformation, z_deformation)

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
plt.imshow(deformed[slice_indices[0], :, :].T)
plt.subplot(2, 3, 5)
plt.imshow(deformed[:, slice_indices[1], :].T)
plt.subplot(2, 3, 6)
plt.imshow(deformed[:, :, slice_indices[2]].T)
plt.show()
