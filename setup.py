# -*- coding: utf-8 -*-

from distutils.core import setup
from glob import glob
import subprocess

scripts = glob('scripts/*')
command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='improc3d',
      version=version,
      description='Useful functions to process 3D images',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Shuo Han',
      url='https://github.com/shuohan/image-processing-3d',
      author_email='shan50@jhu.edu',
      scripts=scripts,
      license='MIT',
      install_requires=['numpy', 'scipy'],
      packages=['improc3d'],
      classifiers=['Programming Language :: Python :: 3',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent']
      )
