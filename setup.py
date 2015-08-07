#!/usr/bin/env python
import os
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

from stormtracks.version import get_version


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='stormtracks',
    version=get_version(),
    description='Tropical Cyclone Detection and Tracking',
    long_description=read('README.rst'),
    author='Mark Muetzelfeldt',
    author_email='markmuetz@gmail.com',
    maintainer='Mark Muetzelfeldt',
    maintainer_email='markmuetz@gmail.com',

    packages=['stormtracks',
              'stormtracks.utils', 
              'stormtracks.processing',
              'stormtracks.analysis',
              'stormtracks.installation',
              'stormtracks.installation.scripts',
              'stormtracks.installation.settings'],
    scripts=[
        'stormtracks/stormtracks-admin.py',
        ],

    ext_modules=[Extension('stormtracks', ['src/cvort.c', 'src/cextrema.c'])],
    install_requires=[
        'pip',
        'ipython',
        'termcolor',
        'argh',
        ],
    package_data={'stormtracks.installation': [
        'requirements/*.txt', 
        'classifiers/*.json', 
        'plots/*.json'
        ]},
    url='https://github.com/markmuetz/stormtracks',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: C',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
    keywords=['tropical cyclone track detection'],
    )
