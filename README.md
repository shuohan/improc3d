# 3D Image Processing Tools

## Introduction

This repository contains some functions to process 3D image.

## Functions

### Rotate 3D image around a point

Example:

```python
import numpy as np
from image_processing import rotate3d
image = np.load(image_path)
# rotate image around x-axis 10 deg, y-axis 20 deg, and z-axis 30 deg
# around the center of the image
rotated = rotate3d(image, 10, 20, 30)
# rotate around point (100, 80, 90)
rotated = rotate3d(image, 10, 20, 30, np.array([100, 80, 90]))
```

### Rotate 3D image around a point

See `tests/scaling.py` for more details.

### Deform 3D image using random elastic deformation

See `tests/deformation.py` for more details.

### Crop 3D image using a mask or bounding box

See `tests/cropping.py` for more details.

## TODO

1. Refactor rotation, scaling, and deformation for duplicated interpolation
