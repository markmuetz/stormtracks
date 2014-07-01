import datetime as dt

import detect as d
import load as l
import match as m
import plotting as pl

num_ensemble_members = 3

tracks, cou = l.load_ibtracks_year(2005)
ncdata = d.NCData(2005)
gdatas = []
all_good_matches = []

for i in range(num_ensemble_members):
    gdata = d.GlobalCyclones(ncdata, i)
    #gdata.track_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 7, 1))
    gdata.track_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 12, 1))
    matches = m.match2(gdata.vort_tracks_by_date, tracks)
    good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]
    gdatas.append(gdata)
    all_good_matches.append(good_matches)
