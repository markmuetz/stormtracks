Release Procedure
=================

* Make sure version.py is at correct version
* Commit changes
* Tag release with e.g. release_0.5.1.0 and push to github.

::

    git commit -a
    git tag release_0.5.1.0
    git push && git push --tags

* [Check credentials in `~/.pypirc`]
* To upload to PyPI, run:

::

    python setup.py sdist upload

* To upload docs, run:

::

    cd docs
    make zip_html

* Then go to https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=stormtracks and upload zip file.
