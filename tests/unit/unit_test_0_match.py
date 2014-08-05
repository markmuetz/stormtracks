import sys
sys.path.insert(0, '..')

import os
import datetime as dt
from glob import glob

from nose.tools import raises
import numpy as np

from stormtracks.load_settings import settings
from stormtracks.matching import EnsembleMatch
from stormtracks.tracking import VortMaxTrack, VortMax


class TestEnsembleMatch:
    def __init__(self):
        self.length = 20
        self.dates = np.array([dt.datetime(2000, 1, i + 1) for i in range(self.length)])
        self.positions = [(i, 0) for i in range(self.length)]
        self.close_positions = [(i, 1) for i in range(self.length)]
        self.far_positions = [(i, 20) for i in range(self.length)]

    def _create_vortmax_track(self, offset, length,  positions):
        vort = 1.

        next_vortmax = None
        for i in range(offset, length)[::-1]:
            vortmax = VortMax(self.dates[i], positions[i], vort)
            if next_vortmax:
                vortmax.next_vortmax.append(next_vortmax)
            next_vortmax = vortmax
        vortmax_track = VortMaxTrack(vortmax)

        return vortmax_track

    def test_1_create_vortmax_track(self):
        vortmax_track = self._create_vortmax_track(0, self.length, self.positions)

        assert len(vortmax_track.vortmaxes) == self.length

    def test_2_cant_add_far_track(self):
        vortmax_track_0 = self._create_vortmax_track(0, self.length, self.positions)
        vortmax_track_1 = self._create_vortmax_track(0, self.length, self.far_positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)

        assert not ensemble_match.add_track(vortmax_track_1)

    def test_3_cant_add_short_track(self):
        vortmax_track_0 = self._create_vortmax_track(0, self.length, self.positions)
        vortmax_track_1 = self._create_vortmax_track(16, self.length, self.positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)

        assert not ensemble_match.add_track(vortmax_track_1)

    def test_4_cant_add_just_overlapping_track(self):
        vortmax_track_0 = self._create_vortmax_track(0, 12, self.positions)
        vortmax_track_1 = self._create_vortmax_track(8, self.length, self.positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)

        assert not ensemble_match.add_track(vortmax_track_1)

    def test_5_can_add_close_track(self):
        vortmax_track_0 = self._create_vortmax_track(0, self.length, self.positions)
        vortmax_track_1 = self._create_vortmax_track(0, self.length, self.close_positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)

        assert ensemble_match.add_track(vortmax_track_1)
        assert (ensemble_match.tracks_added == 2).all()
        assert ensemble_match.cum_dist == 20
        assert ensemble_match.av_vort_track.vortmaxes[0].pos == (0, 0.5)

    def test_6_can_add_close_portion_of_track(self):
        vortmax_track_0 = self._create_vortmax_track(0, self.length, self.positions)
        vortmax_track_1 = self._create_vortmax_track(0, 10, self.close_positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)

        assert ensemble_match.add_track(vortmax_track_1)

        assert len(ensemble_match.tracks_added) == self.length
        assert len(ensemble_match.av_vort_track.vortmaxes) == self.length

        assert (ensemble_match.tracks_added[:10] == 2).all()
        assert (ensemble_match.tracks_added[10:] == 1).all()
        assert ensemble_match.cum_dist == 10
        assert ensemble_match.av_vort_track.vortmaxes[0].pos == (0, 0.5)
        assert ensemble_match.av_vort_track.vortmaxes[10].pos == (10, 0)

    def test_7_complicated_can_add_close_portions_of_track(self):
        vortmax_track_0 = self._create_vortmax_track(0, self.length, self.positions)
        vortmax_track_1 = self._create_vortmax_track(0, 10, self.close_positions)
        vortmax_track_2 = self._create_vortmax_track(5, 15, self.close_positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)

        assert ensemble_match.add_track(vortmax_track_1)
        assert ensemble_match.add_track(vortmax_track_2)

        assert len(ensemble_match.tracks_added) == self.length
        assert len(ensemble_match.av_vort_track.vortmaxes) == self.length

        assert (ensemble_match.tracks_added[:5] == 2).all()
        assert (ensemble_match.tracks_added[5:10] == 3).all()
        assert (ensemble_match.tracks_added[10:15] == 2).all()
        assert (ensemble_match.tracks_added[15:] == 1).all()

        print([vm.pos for vm in ensemble_match.av_vort_track.vortmaxes])
        assert ensemble_match.av_vort_track.vortmaxes[0].pos == (0, 0.5)
        assert ensemble_match.av_vort_track.vortmaxes[9].pos == (9, 2./3.)
        assert ensemble_match.av_vort_track.vortmaxes[10].pos == (10, 0.5)
        assert ensemble_match.av_vort_track.vortmaxes[15].pos == (15, 0)

    def test_8_can_add_longer_track(self):
        vortmax_track_0 = self._create_vortmax_track(5, 15, self.positions)
        vortmax_track_1 = self._create_vortmax_track(0, self.length, self.close_positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)
        assert len(ensemble_match.tracks_added) == 10

        # import ipdb; ipdb.set_trace()
        assert ensemble_match.add_track(vortmax_track_1)

        assert len(ensemble_match.av_vort_track.vortmaxes) == self.length
        assert len(ensemble_match.tracks_added) == self.length

        assert (ensemble_match.tracks_added[:5] == 1).all()
        assert (ensemble_match.tracks_added[5:15] == 2).all()
        assert (ensemble_match.tracks_added[15:] == 1).all()

        print([vm.pos for vm in ensemble_match.av_vort_track.vortmaxes])
        assert ensemble_match.av_vort_track.vortmaxes[0].pos == (0, 1)
        assert ensemble_match.av_vort_track.vortmaxes[10].pos == (10, 1/2.)

    def test_9_can_add_underlapping_track(self):
        vortmax_track_0 = self._create_vortmax_track(5, self.length, self.positions)
        vortmax_track_1 = self._create_vortmax_track(0, 15, self.close_positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)
        assert len(ensemble_match.tracks_added) == 15

        # import ipdb; ipdb.set_trace()
        assert ensemble_match.add_track(vortmax_track_1)

        assert len(ensemble_match.av_vort_track.vortmaxes) == self.length
        assert len(ensemble_match.tracks_added) == self.length

        assert (ensemble_match.tracks_added[:5] == 1).all()
        assert (ensemble_match.tracks_added[5:15] == 2).all()
        assert (ensemble_match.tracks_added[15:] == 1).all()

        print([vm.pos for vm in ensemble_match.av_vort_track.vortmaxes])
        assert ensemble_match.av_vort_track.vortmaxes[0].pos == (0, 1)
        assert ensemble_match.av_vort_track.vortmaxes[10].pos == (10, 1/2.)
        assert ensemble_match.av_vort_track.vortmaxes[15].pos == (15, 0)

    def test_10_can_add_overlapping_track(self):
        vortmax_track_0 = self._create_vortmax_track(0, 15, self.positions)
        vortmax_track_1 = self._create_vortmax_track(5, self.length, self.close_positions)

        ensemble_match = EnsembleMatch(vortmax_track_0)
        assert len(ensemble_match.tracks_added) == 15

        # import ipdb; ipdb.set_trace()
        assert ensemble_match.add_track(vortmax_track_1)

        assert len(ensemble_match.av_vort_track.vortmaxes) == self.length
        assert len(ensemble_match.tracks_added) == self.length

        assert (ensemble_match.tracks_added[:5] == 1).all()
        assert (ensemble_match.tracks_added[5:15] == 2).all()
        assert (ensemble_match.tracks_added[15:] == 1).all()

        print([vm.pos for vm in ensemble_match.av_vort_track.vortmaxes])
        assert ensemble_match.av_vort_track.vortmaxes[0].pos == (0, 0)
        assert ensemble_match.av_vort_track.vortmaxes[10].pos == (10, 1/2.)
        assert ensemble_match.av_vort_track.vortmaxes[15].pos == (15, 1)
