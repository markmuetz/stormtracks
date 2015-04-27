**This project is in Alpha, it might not be easy to get working and the documentation may be lacking details**

The main aim of this project is to develop a procedure to detect and track tropical cyclones in the `C20 Reanalysis Project <http://www.esrl.noaa.gov/psd/data/gridded/data.20thC_ReanV2.html>`_ (20CR). The procedure uses vorticity maxima to locate potential candidates for cyclones, and tracks these maxima using a nearest neighbour approach from time frame to time frame. These derived tracks are then matched to best tracks from the `IBTrACS catalogue <https://climatedataguide.ucar.edu/climate-data/ibtracs-tropical-cyclone-best-track-data>`_. The matching allows the derived tracks to be classified as being either hurricanes or not hurricanes based on their characteristics (such as temp at 850hPa), and using the matched best tracks to train the classifiers. This can be done in a period for which reliable best tracks exist: the satellite era (from 1965 to present day). Once the classifier has been trained, the number of estimated hurricanes can be worked out from just the 20CR data, which can then be used to produce an objective estimate of the historical numbers of hurricanes from earlier time periods. This provides a way of estimating how many hurricanes there were in the period from 1890 to 1965 from the 20CR data, and this can be compared with the best tracks numbers. 

This project was developed for my MSc Dissertation at UCL Geography Department. If you would like a copy of the dissertation please contact me.

Installation
============

The project package is `hosted on PyPI <https://pypi.python.org/pypi?name=stormtracks&:action=display>`_. It can be installed on Debian based systems by following the `installation procedure <http://pythonhosted.org/stormtracks/installation.html>`_.

The code is `hosted on github <https://github.com/markmuetz/stormtracks>`_. Full documentation can be found at `python hosted <http://pythonhosted.org/stormtracks/>`_.
