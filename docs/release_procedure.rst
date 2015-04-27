Release Procedure
=================

* Commit changes
* Tag release with e.g. release_0.3.2 and push to github.
* [Check credentials in `~/.pypirc`]
* Make sure VERSION.txt is at correct version
* To upload to PyPI, run:

    python setup.py sdist upload

* To upload docs, run:

    cd docs
    make zip_html

* Then go to https://pypi.python.org/pypi?%3Aaction=pkg_edit&name=stormtracks and upload zip file.
