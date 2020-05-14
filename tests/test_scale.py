#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.scale import scale3d


parser = argparse.ArgumentParser(description='Test rotation')
parser.add_argument('image', help='The 3D image to rotate')
parser.add_argument('-x', '--x-scale', default=1.5, required=False, type=float,
                    help='The scaling factor along x axis')
parser.add_argument('-y', '--y-scale', default=1.5, required=False, type=float,
                    help='The scaling factor along y axis')
parser.add_argument('-z', '--z-scale', default=1.5, required=False, type=float,
                    help='The scaling factor along z axis')
parser.add_argument('-p', '--point', default=None, required=False, nargs=3,
                    type=float, help='The scaling center')
parser.add_argument('-d', '--duplicated-channels', type=int, default=0,
                    required=False, help='Number of repeats to duplicate the '
                                         'image to multi-channels')
args = parser.parse_args()

image = nib.load(args.image).get_data()
if args.duplicated_channels > 0:
    image = np.repeat(image[None, ...], args.duplicated_channels, 0)

if args.point is not None:
    args.point = np.array(args.point)
scaled = scale3d(image, args.x_scale, args.y_scale, args.z_scale, args.point)
print(scaled.shape)

if args.duplicated_channels > 0:
    image = image[0, ...]
    scaled = scaled[0, ...]

images = (image, scaled)
plt.figure()
for i, im in enumerate(images):
    im = np.transpose(im, axes=[1, 0, 2])
    plt.subplot(len(images), 3, 3 * i + 1)
    plt.imshow(im[:, :, im.shape[2]//2], cmap='gray')
    plt.subplot(len(images), 3, 3 * i + 2)
    plt.imshow(im[:, im.shape[2]//2, :], cmap='gray')
    plt.subplot(len(images), 3, 3 * i + 3)
    plt.imshow(im[im.shape[2]//2, :, :], cmap='gray')
plt.show()
