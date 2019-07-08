Examples
========

Crop
----
.. code-block:: python

    from image_processing_3d import calc_bbox3d, resize_bbox3d, crop3d, uncrop3d

    # load 3D mask and image (using nibabel for example)
    bbox = calc_bbox3d(mask) 
    # target shape can be larger or smaller than the image to crop
    target_bbox_shape = (128, 64, 1000)
    resized_bbox = resize_bbox3d(bbox, target_bbox_shape)
    cropped, source_bbox, target_bbox = crop3d(image, resized_bbox)
    uncropped = uncrop3d(cropped, image.shape, source_bbox, target_bbox)

    from image_processing_3d import padcrop3d
    resized_image, source_bbox, target_bbox = padcrop3d(image, (1000, 100, 200))
    unresized = uncrop3d(resized_image, image.shape, source_bbox, target_bbox)


Reslice (affine transformation)
-------------------------------
.. code-block:: python
   
    import numpy as np
    import nibabel as nib
    from image_processing_3d import reslice3d
    from image_processing_3d import transform_to_axial
    from image_processing_3d import transform_to_coronal
    from image_processing_3d import transform_to_sagittal

    obj = nib.load(image_filename)
    image = obj.get_data()
    affine = obj.affine # affine to LPI- coordinate

    # use interpolation
    axial = transform_to_axial(image, affine, coarse=False)
    coronal = transform_to_coronal(image, affine, coarse=False)
    sagittal = transform_to_sagittal(image, affine, coarse=False)

    LPIm = reslice3d(image, affine)
    # use axis permutation and reflection
    identity = np.eye(4)
    axial_c = transform_to_axial(image, identity, coarse=True)
    coronal_c = transform_to_coronal(image, identity, coarse=True)
    sagittal_c = transform_to_sagittal(image, identity, coarse=True)


Transform
---------
.. code-block:: python

    from image_processing_3d import deform3d, calc_random_deformation3d

    sigma, limit = 8, 5
    # load 3D mask and image (using nibabel for example)
    x_deform = calc_random_deformation3d(image.shape[-3:], sigma, limit)
    y_deform = calc_random_deformation3d(image.shape[-3:], sigma, limit)
    z_deform = calc_random_deformation3d(image.shape[-3:], sigma, limit)
    deformed = deform3d(image, x_deform, y_deform, z_deform)

    from image_processing_3d import rotate3d, scale3d

    rotated = rotate3d(image, 30, -10, 20, pivot=(100, 50, 200))
    scaled = scale3d(image, 2, 0.5, 1.5, pivot=None) # scale around center
