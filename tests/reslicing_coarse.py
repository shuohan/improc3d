#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.reslicing import reslice
from image_processing_3d.reslicing import transform_to_axial
from image_processing_3d.reslicing import transform_to_coronal
from image_processing_3d.reslicing import transform_to_sagittal


obj = nib.load('image2.nii.gz')
image = obj.get_data()
affine = obj.affine
print('original')
print(image.shape)
print(affine)
print('-' * 80)

# image_LPIm = reslice(image, affine)
# axial = transform_to_axial(image_LPIm, np.eye(4), coarse=False)
# coronal = transform_to_coronal(image_LPIm, np.eye(4), coarse=False)
# sagittal = transform_to_sagittal(image_LPIm, np.eye(4), coarse=False)
axial = transform_to_axial(image, affine, coarse=True)
coronal = transform_to_coronal(image, affine, coarse=True)
sagittal = transform_to_sagittal(image, affine, coarse=True)

images = (image, axial, coronal, sagittal)
plt.figure()
for i, im in enumerate(images):
    plt.subplot(len(images), 3, 3 * i + 1)
    plt.imshow(im[:, :, im.shape[2]//2], cmap='gray')
    plt.subplot(len(images), 3, 3 * i + 2)
    plt.imshow(im[:, im.shape[2]//2, :], cmap='gray')
    plt.subplot(len(images), 3, 3 * i + 3)
    plt.imshow(im[im.shape[2]//2, :, :], cmap='gray')
plt.show()
