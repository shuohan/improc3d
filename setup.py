# -*- coding: utf-8 -*-

from distutils.core import setup
from glob import glob
import subprocess

scripts = glob('scripts/*')
command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

setup(name='improc3d',
      version=version,
      description='Useful functions to process 3D images',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      scripts=scripts,
      install_requires=['numpy', 'scipy'],
      packages=['improc3d'])
