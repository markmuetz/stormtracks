import os, sys
sys.path.insert(0, '..')

from glob import glob

from stormtracks.load_settings import settings
from stormtracks.ibtracsdata import IbtracsData


class TestIbtracsDataLocation:
    def test_1_can_find_data(self):
        assert os.path.exists(settings.IBTRACS_DATA_DIR)

    def test_2_file_names_sensible(self):
        file_names = glob(os.path.join(settings.IBTRACS_DATA_DIR, '*.ibtracs.*.nc'))

        for file_name in file_names:
            basename = os.path.basename(file_name)
            split_name = basename.split('.') 
            assert len(split_name) == 4
            assert split_name[1] == 'ibtracs'

    def test_2_check_file_version(self):
        file_names = glob(os.path.join(settings.IBTRACS_DATA_DIR, '*.ibtracs.*.nc'))

        for file_name in file_names:
            basename = os.path.basename(file_name)
            split_name = basename.split('.') 
            assert split_name[2] == 'v03r05'


class TestIbtracsDataLoad:
    def test_1_can_load_2005_data(self):
        ibdata = IbtracsData(settings.IBTRACS_DATA_DIR)
        ibdata.load_ibtracks_year(2005)
        assert len(ibdata.best_tracks) > 30


class TestIbtracsData:
    def setUp(self):
        self.ibdata = IbtracsData(settings.IBTRACS_DATA_DIR)
        self.ibdata.load_ibtracks_year(2005)

    def test_1_test_2005_data(self):
        for best_track in self.ibdata.best_tracks:
            for lon, lat in zip(best_track.lons, best_track.lats):
                print('{0}, {1}'.format(lon, lat))
