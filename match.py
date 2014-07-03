from detect import dist
from collections import defaultdict, OrderedDict

CUM_DIST_CUTOFF = 100

def match(cyclones_by_date, tracks):
    matches = OrderedDict()
    c_set_matches = defaultdict(list)
    unmatched = []
    for track in tracks:
	is_unmatched = True
	for lon, lat, date in zip(track.lon, track.lat, track.dates):
	    if date in cyclones_by_date.keys():
		cyclones = cyclones_by_date[date]
		for c in cyclones:
		    #print(c.cell_pos)
		    #print(lon + 360, lat)
		    #import ipdb; ipdb.set_trace()
		    if dist(c.cell_pos, (lon + 360, lat)) < 10:
			is_unmatched = False
			if not c.cyclone_set in c_set_matches[track]:
			    c_set_matches[track].append(c.cyclone_set)
			matches[(track, date)] = c
	if is_unmatched:
	    unmatched.append(track)
    return matches, c_set_matches, unmatched


class Match(object):
    def __init__(self, track, vort_track):
	self.track = track
	self.vort_track = vort_track
	self.cum_dist = 0
	self.overlap = 1
	self.is_too_far_away = False

    def av_dist(self):
	return self.cum_dist / self.overlap
    

def match2(vort_tracks_by_date, tracks):
    matches = OrderedDict()

    for track in tracks:
	is_unmatched = True
	for lon, lat, date in zip(track.lon, track.lat, track.dates):
	    if date in vort_tracks_by_date.keys():
		vort_tracks = vort_tracks_by_date[date]
		for vortmax in vort_tracks:
		    if (track, vortmax) in matches:
			match = matches[(track, vortmax)]
			match.overlap += 1
		    else:
			match = Match(track, vortmax)
			matches[(track, vortmax)] = match
			if match.is_too_far_away:
			    continue

		    match.cum_dist += dist(vortmax.vortmax_by_date[date].pos, (lon + 360, lat))
		    if match.cum_dist > CUM_DIST_CUTOFF:
			match.is_too_far_away = True

    return matches


def combined_match(best_tracks, all_matches):
    combined_matches = {}

    for best_track in best_tracks:
	combined_matches[best_track] = []

    for matches in all_matches:
	for match in matches:
	    combined_matches[match.track].append(match.vort_track)
    
    return combined_matches

