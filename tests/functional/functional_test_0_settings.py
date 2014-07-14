import os, sys
sys.path.insert(0, '..')

class TestSettings:
    def test_1_can_load_stormtracks_settings(self):
        from stormtracks.load_settings import settings

    def test_2_can_load_stormtracks_pyro_settings(self):
        from stormtracks.load_settings import pyro_settings
