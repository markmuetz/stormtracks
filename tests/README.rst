.. role:: python(code)
   :language: python

Stormtracks testing
===================

Stormtracks tests go in the tests/ directory and subdirectories. Individual bug tests should be named e.g. bugs/bug_21.py where the issue in question is Issue 21 in the `github issue tracker <https://github.com/markmuetz/stormtracks/issues>`_. All tests should be run before each new release.

Tests must be run from the tests/ directory due to the way that the current stormtracks module is loaded (by using :python:`sys.path.insert(0, '..')`)

Run functional tests with:

::

    nosetests functional

or just bugs:

::

    nosetests bugs

or all tests (warning, this may take a while because of interactive/ directory):

::

    nosetests 
