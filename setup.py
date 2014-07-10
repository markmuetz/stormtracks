#!/usr/bin/env python
import os
from distutils.core import setup, Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
      name='stormtracks',
      version='0.1',
      description='Tropical Cyclone Detection and Tracking',
      long_description=read('README.rst'),
      author='Mark Muetzelfeldt',
      author_email='markmuetz@gmail.com',

      packages=['stormtracks', 'stormtracks.settings', 
                'stormtracks.utils', 'stormtracks.pyro_cluster',
                'stormtracks.demo'],
      ext_modules=[Extension('stormtracks', ['src/cvort.c', 'src/cextrema.c'])],
      requires=['numpy', 'scipy', 'pylab', 'mpl_toolkits.Basemap', 'netCDF4'],
      data_files=[('stormtracks/shell_scipts', ['stormtracks/shell_scripts/kill_pyro.sh'])],

      url = 'https://github.com/markmuetz/stormtracks',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Environment :: Console',
          'Intended Audience :: Science/Research',
          'License :: Public Domain',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: C',
          'Topic :: Scientific/Engineering :: Atmospheric Science',
          ],
      keywords=['tropical cyclone track detection'],
     )
