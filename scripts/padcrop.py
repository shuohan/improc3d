#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import nibabel as nib
from improc3d import crop3d, calc_bbox3d, resize_bbox3d, padcrop3d

desc = 'Pad or crop a 3D image to a target shape.'
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('image', help='Image to resize')
parser.add_argument('output', help='Cropped image')
parser.add_argument('-m', '--mask', default=None,
                    help='Mask to calculate the cropping bounding box from')
parser.add_argument('-s', '--target-shape', type=int, nargs=3,
                    default=[192, 192, 192], help='Cropping shape')
args = parser.parse_args()

obj = nib.load(args.image)
image = obj.get_data()

if args.mask is None:
    cropped = padcrop3d(image, args.target_shape)[0]
else:
    mask = nib.load(args.mask).get_data()
    bbox = calc_bbox3d(mask)
    resized_bbox = resize_bbox3d(bbox, args.target_shape)
    cropped = crop3d(image, resized_bbox)[0]

output_obj = nib.Nifti1Image(cropped, obj.affine, obj.header)
output_obj.to_filename(args.output)
