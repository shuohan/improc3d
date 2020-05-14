#!/usr/bin/env bash

../scripts/padcrop.py image.nii.gz crop1.nii.gz -m mask.nii.gz -s 100 100 100
../scripts/padcrop.py image.nii.gz crop2.nii.gz -s 100 100 100
