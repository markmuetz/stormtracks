Installation
============

The recommended way of installing stormtracks is in a `python virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. The instructions below show how to do this on a Debian based Linux system (e.g. Ubuntu, Linux Mint).


Install system dependencies
---------------------------

Installs libraries required to build the python packages (Debian based Linux). Fixes a library so as basemap will build properly by symlinking the required library.

::

    sudo aptitude install git build-essential libhdf5-dev libgeos-dev libproj-dev libfreetype6-dev python-dev libblas-dev liblapack-dev gfortran libnetcdf-dev
    sudo aptitude install python-pip
    sudo pip install virtualenv
    cd /usr/lib/
    sudo ln -s libgeos-3.4.2.so libgeos.so
    cd ~

Create virtualenv
-----------------

Creates and activates a virtualenv within the stormtracks dir (N.B. the virtualenv name is in the .gitignore file).

::

    git clone https://github.com/markmuetz/stormtracks
    cd stormtracks
    virtualenv st_env
    cd st_env
    source bin/activate

Install python packages
-----------------------

This has to be done in two steps. This will build and install all packages (including numpy and scipy), so will take a while. Look at the contents of these two files to see what is required, and basemap requires extra arguments when installing due to how the package is hosted.

::

    pip install -r ../requirements_a.txt
    pip install -r ../requirements_b.txt --allow-external basemap --allow-unverified basemap

Reproduce all figures
---------------------

This will only work if you have all the data required (25GB hosted on dropbox) and your `~.stormtracks/stormtracks_settings.py` file is setup correctly (i.e. based on dotstormtracks.bz2 in the dropbox directory).

::

    python
    >>> from stormtracks import figure_plotting as fp
    >>> fp.main()
