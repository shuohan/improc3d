#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.reslicing import reslice, calc_transformed_shape
from image_processing_3d.reslicing import transform_to_axial
from image_processing_3d.reslicing import transform_to_coronal
from image_processing_3d.reslicing import transform_to_sagittal
from image_processing_3d.cropping import calc_bbox3d


obj = nib.load('image1.nii.gz')
image = obj.get_data()
affine = obj.affine

mask = nib.load('mask1.nii.gz').get_data()
bbox = calc_bbox3d(mask > 0)
pivot = tuple([int((s.stop + s.start) / 2) for s in bbox])

LPIm1 = reslice(image, affine)
shape = calc_transformed_shape(image.shape, affine)
assert np.array_equal(shape, LPIm1.shape)

LPIm2 = reslice(image, affine, target_shape=(100, 100, 100))
LPIm3 = reslice(image, affine, target_shape=(128, 96, 96), pivot_point=pivot)

# axial = transform_to_axial(image, affine)
# coronal = transform_to_coronal(image, affine)
# sagittal = transform_to_sagittal(image, affine)
# 
images = (image, LPIm1, LPIm2, LPIm3)
# images = (image, axial, coronal, sagittal)
plt.figure()
for i, im in enumerate(images):
    im = np.transpose(im, axes=[1, 0, 2])
    plt.subplot(3, len(images), 3 * i + 1)
    plt.imshow(im[:, :, im.shape[2]//2], cmap='gray')
    plt.subplot(3, len(images), 3 * i + 2)
    plt.imshow(im[:, im.shape[1]//2, :], cmap='gray')
    plt.subplot(3, len(images), 3 * i + 3)
    plt.imshow(im[im.shape[0]//2, :, :], cmap='gray')
plt.show()
