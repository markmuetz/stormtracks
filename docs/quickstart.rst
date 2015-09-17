.. _quickstart:

Quickstart
==========

After following the :ref:`installation_pip` guide, go to your installation directory and activate your virtualenv:

::

    source .env/bin/activate

Alternatively, if you used the second :ref:`installation_conda` guide, activate your env:

::
    
    source activate stormtracks

Make sure that `stormtracks_settings.py` has sensible values for the various data directories, and that you have at least 14GB of free disk space. Download 20CR 2005 data and all IBTrACS data, process and analyse 2005. Processing should take around 20 mins, analysis around 2s.

::

    python download_2005.py
    python process_2005.py
    python analyse_2005.py

If everything has worked, you should see output similar to the following after the last command:

::

    (.env)user@machine ~/stormtracks/dev $ python analyse_2005.py 
    vort850, max_ws, pmin, t850
    tp: 14
    tn: 789334
    fp: 3
    fn: 9954
    sens: 0.00140449438202
    ppv : 0.823529411765
    sens*ppv : 0.00115664243225
    Analysed 2005 in 0:00:01.596812

What to do now
--------------

Open an `ipython` shell, and experiment with the three scripts above:

::

    (.env)user@machine ~/stormtracks/dev $ ipython
    Python 2.7.6
    Type "copyright", "credits" or "license" for more information.

    IPython 3.2.1 -- An enhanced Interactive Python.
    ?         -> Introduction and overview of IPython's features.
    %quickref -> Quick reference.
    help      -> Python's own help system.
    object?   -> Details about 'object', use 'object??' for extra details.

    In [1]: from stormtracks.c20data import C20Data

    In [2]: c20data = C20Data(2005)
    2015-08-07 13:48:31,962:st.find_vortmax:INFO    : C20Data: year=2005, version=v1
    2015-08-07 13:48:31,963:st.find_vortmax:INFO    : Using: u9950, v9950, u850, v850, prmsl, t9950, t850, cape, pwat

    In [3]: print c20data.fields
    ['u9950', 'v9950', 'u850', 'v850', 'prmsl', 't9950', 't850', 'cape', 'pwat']

    In [4]: print(c20data.prmsl.shape)
    (56, 91, 180)

    In [5]: c20data.next_date()
    Out[5]: datetime.datetime(2005, 1, 1, 6, 0)
