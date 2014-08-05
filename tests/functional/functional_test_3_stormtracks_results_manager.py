import sys
sys.path.insert(0, '..')
import os

import datetime as dt

from nose.tools import raises

from stormtracks.load_settings import settings
from stormtracks.results import StormtracksResultsManager, ResultNotFound


class TestResultsSave:
    def __init__(self):
        self.year = 1860
        self.bad_year = 1066
        self.ensemble_member = 70
        self.result_key = 'dummy_result'
        self.payload = {'one': 2}

    def setUp(self):
        self.srm = StormtracksResultsManager('_functional_test')

    def test_0_add(self):
        self.srm.add_result(self.year, self.ensemble_member, self.result_key, self.payload)

    def test_1_save(self):
        self.srm.add_result(self.year, self.ensemble_member, self.result_key, self.payload)
        self.srm.save()

    def test_2_get(self):
        result = self.srm.get_result(self.year, self.ensemble_member, self.result_key)
        assert result['one'] == 2

    def test_3_list_year(self):
        assert str(self.year) in self.srm.list_years()

    def test_5_delete(self):
        self.srm.delete(self.year, self.ensemble_member, self.result_key)

    @raises(ResultNotFound)
    def test_6_get_non_existent(self):
        self.srm.get_result(self.bad_year, self.ensemble_member, self.result_key)

    @raises(OSError)
    def test_7_delete_non_existent(self):
        self.srm.delete(self.bad_year, self.ensemble_member, self.result_key)
