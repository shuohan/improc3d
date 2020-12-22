#!/usr/bin/env python

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from improc3d.rotate import rotate3d


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
parser.add_argument('-d', '--duplicated-channels', type=int, default=0,
                    required=False, help='Number of repeats to duplicate the '
                                         'image to multi-channels')
args = parser.parse_args()

image = nib.load(args.image).get_data()
if args.duplicated_channels > 0:
    image = np.repeat(image[None, ...], args.duplicated_channels, 0)

if args.point is not None:
    args.point = np.array(args.point)
rotated = rotate3d(image, args.x_angle, args.y_angle, args.z_angle, args.point)

if args.duplicated_channels == 0:
    image = image[None, ...]
    rotated = rotated[None, ...]

for im, r in zip(image, rotated):

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
    plt.imshow(r[slice_indices[0], :, :].T)
    plt.subplot(2, 3, 5)
    plt.imshow(r[:, slice_indices[1], :].T)
    plt.subplot(2, 3, 6)
    plt.imshow(r[:, :, slice_indices[2]].T)

plt.show()
