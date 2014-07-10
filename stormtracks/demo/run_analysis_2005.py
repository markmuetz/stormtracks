import datetime as dt

import pylab as plt

from stormtracks.c20data import C20Data
from stormtracks.ibtracsdata import IbtracsData
from stormtracks.detect import GlobalCyclones
from stormtracks.match import match2

import stormtracks.plotting as pl

# Create a wrapper for the C20 Reanalysis data.
c20data = C20Data(2005)
c20data.first_date()
# Plot PSL for 1st of Jan 2005.
pl.plot_on_earth(c20data.lon, c20data.lat, c20data.psl)
plt.show()

# Plot vorticity for 1st of Jan 2005.
pl.plot_on_earth(c20data.lon, c20data.lat, c20data.vort)
plt.show()

# Load IBTrACS data for 2005.
ibtracs = IbtracsData()
best_tracks = ibtracs.load_ibtracks_year(2005)

# Run the analysis for 2005.
gdata = GlobalCyclones(c20data, ensemble_member=0)
gdata.track_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 7, 1))

# Match the generated tracks agains the best tracks.
matches = match2(gdata.vort_tracks_by_date, best_tracks)
good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]

for gm in good_matches:
    pl.plot_match(gm, None)
    plt.show()
