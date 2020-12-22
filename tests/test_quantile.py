#!/usr/bin/env python

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from improc3d.intensity import quantile_scale


image = nib.load('image.nii.gz').get_data()
scaled = quantile_scale(image, lower_th=50, upper_th=100)
print('orig min', np.min(image), 'orig max', np.max(image))
print('scaled min', np.min(scaled), 'scaled max', np.max(scaled))

plt.figure()
plt.subplot(121)
plt.imshow(image[:, :, image.shape[-1]//2], cmap='gray')
plt.subplot(122)
plt.imshow(scaled[:, :, image.shape[-1]//2], cmap='gray')
plt.show()
