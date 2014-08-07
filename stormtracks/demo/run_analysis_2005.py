import datetime as dt

import pylab as plt

from stormtracks.c20data import C20Data, GlobalEnsembleMember
from stormtracks.ibtracsdata import IbtracsData
from stormtracks.matching import match_vort_tracks_by_date_to_best_tracks, good_matches
from stormtracks.tracking import VortmaxFinder, VortmaxNearestNeighbourTracker

import stormtracks.plotting as pl

# Create a wrapper for the C20 Reanalysis data.
c20data = C20Data(2005)
c20data.first_date()
# Plot PSL for 1st of Jan 2005.
pl.raster_on_earth(c20data.lons, c20data.lats, c20data.psl)
plt.show()

# Plot vorticity for 1st of Jan 2005.
pl.raster_on_earth(c20data.lons, c20data.lats, c20data.vort)
plt.show()

# Load IBTrACS data for 2005.
ibtracs = IbtracsData()
best_tracks = ibtracs.load_ibtracks_year(2005)

# Run the analysis for 2005.
gdata = GlobalEnsembleMember(c20data, ensemble_member=0)

vort_finder = VortmaxFinder(gdata)
vort_finder.find_vort_maxima(dt.datetime(2005, 6, 1), dt.datetime(2005, 7, 1))
tracker = VortmaxNearestNeighbourTracker()
tracker.track_vort_maxima(vort_finder.vortmax_time_series)

# Match the generated tracks agains the best tracks.
matches = match_vort_tracks_by_date_to_best_tracks(tracker.vort_tracks_by_date, best_tracks)
gms = good_matches(matches)

for gm in gms:
    pl.plot_match_with_date(gm, None)
    plt.show()
