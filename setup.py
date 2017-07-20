#!/usr/bin/env python

from distutils.core import setup
from pip.req import parse_requirements

setup(name='tdxformula',
      version='1.0.0',
      description='tdx formula executor',
      install_requires=[str(ir.req) for ir in parse_requirements("requirements.txt", session=False)],
      author='Woody',
      author_email='li.woodyli@gmail.com',
      url='https://github.com/woodylee1974/tdxformula',
      packages=['tdx'],
     )