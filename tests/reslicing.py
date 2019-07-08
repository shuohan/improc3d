#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# from image_processing_3d.reslicing import transform_to_axial
# from image_processing_3d.reslicing import transform_to_coronal
# from image_processing_3d.reslicing import transform_to_sagittal

from image_processing_3d.reslicing import reslice


obj = nib.load('image2.nii.gz')
image = obj.get_data()
affine = obj.affine
print(affine)
# print('original')
# print(image.shape)
# print(affine)
# print('-' * 80)

result = reslice(image, affine)

# axial = transform_to_axial(image, affine)
# coronal = transform_to_coronal(image, affine)
# sagittal = transform_to_sagittal(image, affine)

# images = (image, axial, coronal, sagittal)
images = (image, result)
plt.figure()
for i, im in enumerate(images):
    plt.subplot(len(images), 3, 3 * i + 1)
    plt.imshow(im[:, :, im.shape[2]//2], cmap='gray')
    plt.subplot(len(images), 3, 3 * i + 2)
    plt.imshow(im[:, im.shape[2]//2, :], cmap='gray')
    plt.subplot(len(images), 3, 3 * i + 3)
    plt.imshow(im[im.shape[2]//2, :, :], cmap='gray')
plt.show()
