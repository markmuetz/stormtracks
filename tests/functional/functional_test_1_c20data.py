import sys
sys.path.insert(0, '..')

import os

from glob import glob

from nose.tools import raises

from stormtracks.load_settings import settings
from stormtracks.c20data import C20Data


class TestC20DataLocation:
    def test_1_can_find_data(self):
        assert os.path.exists(settings.C20_FULL_DATA_DIR)

        c20_full_data_dirs = glob(os.path.join(settings.C20_FULL_DATA_DIR, '*'))
        assert len(c20_full_data_dirs) != 0

    def test_2_dir_name_sensible(self):
        c20_full_data_dirs = glob(os.path.join(settings.C20_FULL_DATA_DIR, '*'))
        dir_name = os.path.basename(c20_full_data_dirs[0])
        year = int(dir_name)
        assert 1870 < year < 2014


class TestC20DataLoad:
    def test_1_can_load_data(self):
        c20_full_data_dirs = glob(os.path.join(settings.C20_FULL_DATA_DIR, '*'))
        c20data = C20Data(os.path.basename(c20_full_data_dirs[0]))


class TestC20DataAnalyse:
    def setUp(self):
        c20_full_data_dirs = glob(os.path.join(settings.C20_FULL_DATA_DIR, '*'))
        c20data = C20Data(os.path.basename(c20_full_data_dirs[0]))
        self.c20data = c20data

    def test_1_first_date(self):
        self.c20data.first_date()

    def test_2_psl_shape(self):
        self.c20data.first_date()
        assert self.c20data.psl.shape == (91, 180)

    def test_3_psl_sum(self):
        self.c20data.first_date()
        psl_sum = self.c20data.psl.sum()

    def test_4_vort_shape(self):
        self.c20data.first_date()
        assert self.c20data.vort.shape == (91, 180)

    def test_5_vort_sum(self):
        self.c20data.first_date()
        vort_sum = self.c20data.vort.sum()

    def test_6_vort_shape(self):
        self.c20data.first_date()
        assert self.c20data.vort4.shape == (91, 180)

    def test_7_vort_sum(self):
        self.c20data.first_date()
        vort_sum = self.c20data.vort4.sum()

    @raises(AttributeError)
    def test_8_no_smoothed(self):
        self.c20data.first_date()
        self.c20data.smoothed_vort

    def test_9_smoothed_vort_sum(self):
        self.c20data.smoothing = True
        self.c20data.first_date()
        smoothed_vort_sum = self.c20data.smoothed_vort.sum()

    def tearDown(self):
        self.c20data.close_datasets()
