import os, sys
sys.path.insert(0, '..')

import datetime as dt

from stormtracks.plotting import Plotter
import mock 

class TestPlotterLoadSave:
    def __init__(self):
        self.test_plotter_settings_name = 'test_bug_22'

    def setUp(self):
        c20data = mock.Mock()
        c20data.first_date.return_value = dt.datetime(2001, 1, 1)
        self.plotter = Plotter(None, c20data, None, None)

    def test_1_save(self):
        self.plotter.save(self.test_plotter_settings_name)
        
    def test_2_save_load(self):
        self.plotter.save(self.test_plotter_settings_name)
        self.plotter.load(self.test_plotter_settings_name)

    def test_3_save_list(self):
        self.plotter.save(self.test_plotter_settings_name)
        assert self.test_plotter_settings_name in self.plotter.list()

    def test_4_delete(self):
        self.plotter.delete(self.test_plotter_settings_name)
