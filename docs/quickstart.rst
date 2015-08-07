.. _quickstart:

Quickstart
==========

After following the :ref:`installation` guide, go to your installation directory and activate your virtualenv:

::

    source .env/bin/activate

Download 20CR 2005 data and all IBTrACS data, process and analyse 2005:

::

    python download_2005.py
    python process_2005.py
    python analyse_2005.py
