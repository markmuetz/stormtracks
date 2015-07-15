#!/usr/bin/env python
import os
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension


def home_dir():
    return os.path.expandvars('$HOME')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='stormtracks',
    version=read('VERSION.txt').rstrip(),
    description='Tropical Cyclone Detection and Tracking',
    long_description=read('README.rst'),
    author='Mark Muetzelfeldt',
    author_email='markmuetz@gmail.com',
    maintainer='Mark Muetzelfeldt',
    maintainer_email='markmuetz@gmail.com',

    packages=['stormtracks', 'stormtracks.settings',
              'stormtracks.utils', 
              #'stormtracks.pyro_cluster',
              'stormtracks.demo'],
    scripts=[
        'stormtracks/stormtracks-admin.py',
        ],

    ext_modules=[Extension('stormtracks', ['src/cvort.c', 'src/cextrema.c'])],
    install_requires=[
        'pip',
        'ipython',
        ],
#    data_files=[
#        ('', ['shell_scripts/kill_pyro_worker.sh']),
#        (os.path.join(home_dir(), '.stormtracks'),
#            ['stormtracks/settings/default_stormtracks_settings.py']),
#        (os.path.join(home_dir(), '.stormtracks'),
#            ['stormtracks/settings/default_stormtracks_pyro_settings.py']),
#        ],
#
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
