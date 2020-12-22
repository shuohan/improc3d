#!/usr/bin/env python

import sys
sys.path.insert(0, '..')

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.translate import translate3d_int 


parser = argparse.ArgumentParser(description='Test rotation')
parser.add_argument('image', help='The 3D image to rotate')
parser.add_argument('-x', '--x-translation', default=5, required=False,
                    type=float, help='The translation along x axis')
parser.add_argument('-y', '--y-translation', default=5, required=False,
                    type=float, help='The translation along y axis')
parser.add_argument('-z', '--z-translation', default=5, required=False,
                    type=float, help='The rotation angle along z axis')
parser.add_argument('-d', '--duplicated-channels', type=int, default=0,
                    required=False, help='Number of repeats to duplicate the '
                                         'image to multi-channels')
args = parser.parse_args()

image = nib.load(args.image).get_data()
if args.duplicated_channels > 0:
    image = np.repeat(image[None, ...], args.duplicated_channels, 0)

translated = translate3d_int(image, args.x_translation, args.y_translation,
                             args.z_translation)

if args.duplicated_channels == 0:
    image = image[None, ...]
    translated = translated[None, ...]

for im, r in zip(image, translated):

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
