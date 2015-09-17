.. _installation_pip:

Installation using pip
======================

One way of installing stormtracks is using `pip <https://pip.pypa.io/en/stable/>`_  in a `python virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. The instructions below show how to do this on a Debian based Linux system (e.g. Ubuntu, Linux Mint) and a recent Fedora Core system (Fedora Core 22). They should be usable with minor modifications on other Linux/Unix platforms, e.g. on older Fedora Core system replace `dnf` with `yum`.


Install system dependencies (Debian)
------------------------------------

Open a terminal and run these commands to get pip, virtualenv and some tools for compiling binaries.

::

    sudo aptitude install python-pip build-essential
    sudo pip install virtualenv

Install system dependencies (Fedora Core 22)
--------------------------------------------

Open a terminal and run these commands to get pip, virtualenv and some tools for compiling binaries.

::

    sudo dnf install gcc python-devel
    sudo pip install virtualenv

Create virtualenv
-----------------

Creates and activates a virtualenv to run stormtracks from:

::

    mkdir stormtracks
    cd stormtracks
    virtualenv .env
    source .env/bin/activate


Install stormtracks
-------------------

Installing stormtracks using pip will install `stormtracks-admin.py`, a utility which can help install stormtrack's dependencies. This will perform a complete install on Debian based linux computers. This will prompt you for your root password, if you are unhappy about this then use the next manual method to see what is going on. First installation will take a while, around 20-30 mins. Subsequent installations should be faster, around 1 min.

::

    pip install stormtracks # N.B. relies on gcc/make being installed.
    stormtracks-admin.py install-full
    # stormtracks-admin.py install-full -o fedora_core
    

(Alternative) Manually install system dependencies
--------------------------------------------------

Installs libraries required to build the python packages (Debian based Linux). Fixes a library so as basemap will build properly by symlinking the required library. Installs pip requirements in 4 stages to get round some installation problems with some of the modules.

::

    stormtracks-admin.py print-installation-commands
    # stormtracks-admin.py print-installation-commands -o fedora_core
    stormtracks-admin.py print-installation-commands > install.sh
    bash install.sh

.. _installation_conda:

Installation using conda
========================

`Conda <http://conda.pydata.org/docs/>`_ can also be used to install most of the packages. (Conda is roughly the equivalent of pip and virtualenv combined.) The general idea is the same as for pip, although due to the difference in the way packages are bundled this might be (far) faster as no compilation of software is required. First, install conda, which can be done by downloading and installing `anaconda <https://store.continuum.io/cshop/anaconda/>`_.

::

    mkdir stormtracks
    cd stormtracks
    conda create -n stormtracks python # Creates conda env.
    source activate stormtracks # Activates env.
    pip install stormtracks # N.B. relies on gcc/make being installed.
    stormtracks-admin.py copy-files # Copies files to current dir.
    conda install --file requirements/conda_requirements.txt # Installs most dependencies.
    pip install -r requirements/conda_pip_requirements.txt # Installs simplejson.
    

Where to go from here
=====================

Head to the :ref:`quickstart` guide to see how to download, process and analyse one year.
