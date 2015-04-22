#!/usr/bin/env python
import os
from distutils.core import setup, Extension


def home_dir():
    return os.path.expandvars('$HOME')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='stormtracks',
    version='0.3.1',
    description='Tropical Cyclone Detection and Tracking',
    long_description=read('README.rst'),
    author='Mark Muetzelfeldt',
    author_email='markmuetz@gmail.com',
    maintainer='Mark Muetzelfeldt',
    maintainer_email='markmuetz@gmail.com',

    packages=['stormtracks', 'stormtracks.settings',
              'stormtracks.utils', 'stormtracks.pyro_cluster',
              'stormtracks.demo'],
    scripts=[
        'stormtracks/run_stormtracks.py',
        'stormtracks/pyro_cluster/pyro_nameserver.py',
        'stormtracks/pyro_cluster/pyro_starter.py',
        'stormtracks/pyro_cluster/pyro_manager.py',
        'stormtracks/pyro_cluster/pyro_worker.py',
        'stormtracks/pyro_cluster/pyro_ender.py',
        ],

    ext_modules=[Extension('stormtracks', ['src/cvort.c', 'src/cextrema.c'])],
    requires=['numpy', 'scipy', 'pylab', 'mpl_toolkits.Basemap', 'netCDF4', 'simplejson'],
    data_files=[
        (os.path.join(home_dir(), '.stormtracks/shell_scripts'),
            ['shell_scripts/kill_pyro_worker.sh']),
        (os.path.join(home_dir(), '.stormtracks'),
            ['stormtracks/settings/default_stormtracks_settings.py']),
        (os.path.join(home_dir(), '.stormtracks'),
            ['stormtracks/settings/default_stormtracks_pyro_settings.py']),
        ],

    url='http://markmuetz.github.io/stormtracks/',
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
