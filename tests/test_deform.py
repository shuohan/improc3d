#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.deform import deform3d, calc_random_deformation3d


parser = argparse.ArgumentParser(description='Test random free deformation')
parser.add_argument('image', help='The 3D image to deform')
parser.add_argument('-s', '--sigma', default=3, required=False, type=float,
                    help='Control the smoothness of the deformation field')
parser.add_argument('-l', '--limit', default=3, required=False, type=float,
                    help='The maximum possible deformation in pixel')
parser.add_argument('-d', '--duplicated-channels', type=int, default=0,
                    required=False, help='Number of repeats to duplicate the '
                                         'image to multi-channels')
args = parser.parse_args()

image = nib.load(args.image).get_data()
if args.duplicated_channels > 0:
    image = np.repeat(image[None, ...], args.duplicated_channels, 0)

x_deformation = calc_random_deformation3d(image.shape[-3:], args.sigma,
                                          args.limit)
y_deformation = calc_random_deformation3d(image.shape[-3:], args.sigma,
                                          args.limit)
z_deformation = calc_random_deformation3d(image.shape[-3:], args.sigma,
                                          args.limit)
deformed = deform3d(image, x_deformation, y_deformation, z_deformation)

if args.duplicated_channels == 0:
    image = image[None, ...]
    deformed = deformed[None, ...]

for im, d in zip(image, deformed):

    shape = im.shape
    slice_indices = np.array(shape) // 2
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(im[slice_indices[0], :, :].T)
    plt.subplot(2, 3, 2)
    plt.imshow(im[:, slice_indices[1], :].T)
    plt.subplot(2, 3, 3)
    plt.imshow(im[:, :, slice_indices[2]].T)
    plt.subplot(2, 3, 4)
    plt.imshow(d[slice_indices[0], :, :].T)
    plt.subplot(2, 3, 5)
    plt.imshow(d[:, slice_indices[1], :].T)
    plt.subplot(2, 3, 6)
    plt.imshow(d[:, :, slice_indices[2]].T)

plt.show()
