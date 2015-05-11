stormtracks tests go in the tests/ directory and subdirectories. Individual bug tests should be named e.g. bugs/test_bug_22.py where the issue in question is Issue 22 in the `github issue tracker <https://github.com/markmuetz/stormtracks/issues>`_. All tests should be run before each new release. Additionally, style checking is done using the pep8 `style checker <https://pypi.python.org/pypi/pep8>`_, with a modified maximum line length of 100 characters.

Tests must be run from the tests/ directory due to the way that the current stormtracks module is loaded (by using ``sys.path.insert(0, '../..')``)

Run functional tests with:

::

    nosetests functional

`PEP8 <http://legacy.python.org/dev/peps/pep-0008/>`_ tests with:

::

    nosetests pep8_tests

or just bugs:

::

    nosetests bugs

or all tests (warning, this may take a while because of interactive tests, which will show matplotlib displays):

::

    nosetests 
