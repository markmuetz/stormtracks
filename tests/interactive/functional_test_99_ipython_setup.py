import os, sys
sys.path.insert(0, '..')

class TestIpythonSetup:
    def test_1_load_module(self):
        import stormtracks.ipython_setup as ipython_setup
