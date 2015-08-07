.. _installation:

Installation
============

The recommended way of installing stormtracks is in a `python virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_. The instructions below show how to do this on a Debian based Linux system (e.g. Ubuntu, Linux Mint).


Install system dependencies
---------------------------

::

    sudo aptitude install python-pip build-essential
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

Installing stormtracks using pip will install `stormtracks-admin.py`, a utility which can help install stormtrack's dependencies. This will perform a complete install on Debian based linux computers. This will prompt you for your root password, if you are unhappy about this use the next manual method.

::

    pip install stormtracks
    stormtracks-admin.py install-full
    

(Alternative) Manually install system dependencies
--------------------------------------------------

Installs libraries required to build the python packages (Debian based Linux). Fixes a library so as basemap will build properly by symlinking the required library. Installs pip requirements in 4 stages to get round some installation problems with some of the modules.

::

    stormtracks-admin.py print-installation-commands
    stormtracks-admin.py print-installation-commands > install.sh
    bash install.sh

Where to go from here
---------------------

Head to the :ref:`quickstart` guide to see how to download, process and analyse one year.
