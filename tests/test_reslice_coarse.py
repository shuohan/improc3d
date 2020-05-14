#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.reslice import reslice3d, reslice3d_coarse
from image_processing_3d.reslice import transform_to_axial
from image_processing_3d.reslice import transform_to_coronal
from image_processing_3d.reslice import transform_to_sagittal


obj = nib.load('image1.nii.gz')
image = obj.get_data()
affine = obj.affine
print(image.shape)
print(np.round(affine))

axial_c = transform_to_axial(image, affine, coarse=True)
coronal_c = transform_to_coronal(image, affine, coarse=True)
sagittal_c = transform_to_sagittal(image, affine, coarse=True)

LPIm = reslice3d(image, affine)
axial = transform_to_axial(LPIm, np.eye(4), coarse=True)
coronal = transform_to_coronal(LPIm, np.eye(4), coarse=True)
sagittal = transform_to_sagittal(LPIm, np.eye(4), coarse=True)

images = (image, axial_c, axial, coronal_c, coronal, sagittal_c, sagittal)
plt.figure()
for i, im in enumerate(images):
    im = np.transpose(im, axes=[1, 0, 2])
    plt.subplot(3, len(images), len(images) * 0 + i + 1)
    plt.imshow(im[:, :, im.shape[2]//2], cmap='gray')
    plt.subplot(3, len(images), len(images) * 1 + i + 1)
    plt.imshow(im[:, im.shape[1]//2, :], cmap='gray')
    plt.subplot(3, len(images), len(images) * 2 + i + 1)
    plt.imshow(im[im.shape[0]//2, :, :], cmap='gray')
plt.show()
