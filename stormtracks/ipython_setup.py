# Puts some interesting objects in an ipython interactive shell
# Should be run using e.g.:
# 
# In [1]: run ipython_setup.py
import sys
import time
import datetime as dt
import socket

import numpy as np
import pylab as plt

import detect as d
import match as m
import plotting as pl
import utils.kalman as k
from ibtracsdata import IbtracsData

#num_ensemble_members = 56
num_ensemble_members = 3

start = time.time()
print(start)

short_name = socket.gethostname().split('.')[0] 

if short_name == 'linz':
    ensemble_member_range = range(0, 3)
elif short_name == 'athens':
    sys.exit('bad computer')
    ensemble_member_range = range(3, 6)
elif short_name == 'madrid':
    ensemble_member_range = range(6, 9)
elif short_name == 'warsaw':
    ensemble_member_range = range(9, 12)
elif short_name == 'prague':
    ensemble_member_range = range(12, 15)
elif short_name == 'berlin':
    ensemble_member_range = range(15, 18)
elif short_name == 'determinist-mint':
    ensemble_member_range = range(0, 1)
else:
    ensemble_member_range = range(0, num_ensemble_members)

itd = IbtracsData()
tracks = itd.load_ibtracks_year(2005)
c20data = d.C20Data(2005, verbose=False)
gdatas = []
all_good_matches = []

for i in ensemble_member_range:
    print('Ensemble member {0} of {1}'.format(i + 1, len(ensemble_member_range)))
    gdata = d.GlobalCyclones(c20data, i)
    #gdata.track_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 7, 1))
    gdata.track_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 12, 1))
    matches = m.match2(gdata.vort_tracks_by_date, tracks)
    good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]
    gdatas.append(gdata)
    all_good_matches.append(good_matches)

end = time.time()

combined_matches = m.combined_match(tracks, all_good_matches)

if True:
    gm2 = good_matches[2]
    pos = np.array([vm.pos for vm in gm2.vort_track.vortmaxes])
    x, P, y = k.demo_simple_2d_with_inertia(pos[0], pos, 1e-1, 1e1)
    plt.clf()
    plt.plot(gm2.track.lon + 360, gm2.track.lat, 'r-')
    plt.plot(pos[:, 0], pos[:, 1], 'k+')
    plt.plot(x[:, 0], x[:, 1])

print('{0} - {1}'.format(short_name, ensemble_member_range))
print('Start: {0}, end: {1}, duration: {2}'.format(start, end, end - start))
