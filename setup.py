# -*- coding: utf-8 -*-

from distutils.core import setup
from glob import glob
import subprocess

scripts = glob('scripts/*')
command = ['git', 'describe', '--tags']
tag = subprocess.check_output(command).decode().strip()
head_hash = ['git', 'show-ref', '--head', 'HEAD']
tag_hash = ['git', 'show-ref', '--tags', tag]
version = tag if tag_hash == head_hash else head_hash

setup(name='image-processing-3d',
      version=version,
      description='Useful functions to process 3D images',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      packages=['image_processing_3d'])
