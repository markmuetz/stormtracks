Change Log
==========

Version 0.4 (Alpha) - Planned for April 28, 2015
-------------------------------------

* Tidy up code
* Improve installation procedure
* Work on reproducibility (through virtualenv)

Version MSc_Dissertation (Alpha) - August 28, 2014
--------------------------------------------------

* Implementing SGD training/matching against IBTrACS data
* Look into different classifiers: SGD/QDC/cutoff etc.
* Plot all figures for dissertation

Version 0.3 (Alpha) - August 15, 2014
-------------------------------------

* Implementing SGD training/matching against IBTrACS data
* Improve worker crash handling in pyro_cluster
* Improve documentation
* Run for 20 years worth of analysis on UCL computers
* Fix bugs with Pyro code
* Fix up tests

Version 0.2 (Alpha) - July 18, 2014
-----------------------------------

* Abstract out tracking code and have first shot at implementing a Kalman filter based tracker
* Implement StormtracksResultsManager that can be used for saving/loading results to disk
* Improve pyro_cluster code to the point where it is usable (speedups of 10x using 15 UCL computers)
* Add logging to pyro_cluster code to aid debugging
* Handle lons/lats consistently - always use -180 to 180 in code
* Add some functional/bug tests
* PEP8 test all code in tests
* Document most classes/functions (and add a coverage check in documentation)
* Make names more consistent in code
* Fix some bugs

Version 0.1 (Alpha) - July 10, 2014
-----------------------------------

* Download and load of IBTrACS and C20 Reanalysis data
* Vorticity tracking (nearest neighbour) in place
* Matching of vorticity to best tracks
* Plotting of output on world maps
* pyro_cluster proof of concept working
* Make installable through PyPI
* Add structure for documentation
* Proper settings file that is easy to modify and stored in ~/.stormtracks

Version 0.0.X
-------------

* Experimented with a range of ways of tracking tropical cyclone:
    * Adapting Qinling Wu's eddy detection algorithm
        * Used a modified voronoi segmentation scheme to partition up ocean
        * Then used a region shirinking technique to localise and detect eddy centres
        * Made sure that the Okubu-Weiss paramater was satisfied for the eddy centres
        * It wasn't overly suitable for tracking hurricanes:
        * There are typically ~1000s of eddies in the ocean, whereas there will only
          be ~ 100s of pressure minima
    * Through min pressure:
        * Found to be hard to track pressure minima from one timestep to the other
        * After talking to Kevin Hodges (Reading University) this approach was abandoned *for tracking*
        * However lot of the code developed (e.g. finding Radius of Outermost Closed Isobar - ROCI) may still be useful
        * Code is located in cyclone.py
    * Settled on tracking vorticity maxima
* Look at Kevin Walsh's Fortran algorithm for cyclone tracking and take some ideas (and an implementation of a 4th order vorticity algorithm)
    * Ran algorithm against C20 data and found Katrina/Wilma in 2005 data
* Get Kevin Hodges TRACK code building (but don't manage to get it running against data)
* Downloaded C20 u9950/v9950, prmsl fields for vorticity/pressure
    * Full data sets used, i.e. that contain each ensemble member separately
    * This is better than the mean for tracking features (according to Chris Brierley and Kevin Hodges)
    * Kevin Hodges recommended using an average of wind fields
* Plot a variety of different data fields to get a feel for data:
    * e.g. vorticity with best track overlayed
* Speed up some of the analysis:
    * c functions for vorticity calculations
* Basic analysis:
    * pressure min, vorticity max etc.
* Look into parallelising analysis:
    * Settled on Pyro4 library for remote python execution
    * Set up a basic manager/worker system and tested on UCL computers
* Experiment with Support Vector Machine (SVM) implementation in sckikit-learn
* Implement a basic Kalman filter and check that it is producing reasonable data
* Smoothing and upscaling of vorticity/pressure data
