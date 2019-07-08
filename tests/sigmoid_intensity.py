#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from image_processing_3d.intensity import calc_random_intensity_transform


low = -0.5
high = 0.5
transform = calc_random_intensity_transform()
x = np.linspace(low, high, 100)
y = transform(x)
plt.plot(x, y)
plt.xlim(low, high)
plt.ylim(low, high)

image = 'image.nii.gz'
image = nib.load(image).get_data()
shape = image.shape
new_image = transform(image)

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(image[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(image[:, :, shape[2]//2], cmap='gray')

plt.subplot(2, 3, 4)
plt.imshow(new_image[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(new_image[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(new_image[:, :, shape[2]//2], cmap='gray')

plt.show()
