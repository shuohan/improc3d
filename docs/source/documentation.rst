.. include:: text_colors.rst

Documentation
=============

.. automodule:: image_processing_3d

Resize
------
.. autofunction:: crop3d
.. autofunction:: uncrop3d
.. autofunction:: calc_bbox3d
.. autofunction:: resize_bbox3d
.. autofunction:: padcrop3d

Reslice
-------
.. automodule:: image_processing_3d.reslice
.. automodule:: image_processing_3d
   :noindex: 
.. autofunction:: reslice3d
.. autofunction:: reslice3d_coarse
.. autofunction:: transform_to_axial
.. autofunction:: transform_to_coronal
.. autofunction:: transform_to_sagittal
.. autofunction:: calc_transformed_shape
.. autofunction:: convert_LPIm_to_RAIm
.. autofunction:: convert_LPIm_to_RSAm
.. autofunction:: convert_LPIm_to_ASRm

Transform
---------
.. autofunction:: deform3d
.. autofunction:: rotate3d
.. autofunction:: scale3d
.. autofunction:: translate3d_int
.. autofunction:: calc_random_deformation3d

Intensity
---------
.. autofunction:: quantile_scale
.. autofunction:: calc_random_intensity_transform

Utilities
-----------------------
.. automodule:: image_processing_3d.utils
   :members:
