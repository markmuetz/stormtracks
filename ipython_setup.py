import time
import datetime as dt
import socket

import detect as d
import load as l
import match as m
import plotting as pl

#num_ensemble_members = 56
num_ensemble_members = 3

start = time.time()
print(start)

short_name = socket.gethostname().split('.')[0] 

if short_name == 'linz':
    ensemble_member_range = range(0, 3)
elif short_name == 'athens':
    ensemble_member_range = range(3, 6)
elif short_name == 'madrid':
    ensemble_member_range = range(6, 9)
elif short_name == 'madrid':
    ensemble_member_range = range(6, 9)
elif short_name == 'determinist-mint':
    ensemble_member_range = range(9, 12)

tracks, cou = l.load_ibtracks_year(2005)
ncdata = d.NCData(2005, verbose=False)
gdatas = []
all_good_matches = []

for i in ensemble_member_range:
    gdata = d.GlobalCyclones(ncdata, i)
    #gdata.track_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 7, 1))
    gdata.track_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 12, 1))
    matches = m.match2(gdata.vort_tracks_by_date, tracks)
    good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]
    gdatas.append(gdata)
    all_good_matches.append(good_matches)

end = time.time()

print('{0} - {1}'.format(short_name, ensemble_member_range))
print('Start: {0}, end: {1}, duration: {2}'.format(start, end, end - start))
