import datetime as dt

import detect as d
import load as l
import match as m
import plotting as pl

ncdata = d.NCData(2005)
gdata = d.GlobalCyclones(ncdata)
#gdata.track_vort_maxima(0, dt.datetime(2005, 6, 1), dt.datetime(2005, 7, 1))
gdata.track_vort_maxima(0, dt.datetime(2005, 6, 1), dt.datetime(2005, 12, 1))
gdata.construct_vortmax_tracks_by_date()
tracks, cou = l.load_ibtracks_year(2005)
matches = m.match2(gdata.vort_tracks_by_date, tracks)
good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]
