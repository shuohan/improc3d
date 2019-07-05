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
    cropped, source_bbox, target_bbx = crop3d(image, resized_bbox)
    uncropped = uncrop3d(cropped, image.shape, source_bbox, target_bbox)

    from image_processing_3d import padcrop3d
    resized_image, source_bbox, target_bbox = padcrop3d(image, (1000, 100, 200))
    unresized = uncrop3d(resized_image, image.shape, source_bbox, target_bbox)


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

    rotated = rotate3d(image, 30, -10, 20, point=(100, 50, 200))
    scaled = scale3d(image, 2, 0.5, 1.5, point=None) # scale around center
