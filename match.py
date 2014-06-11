from detect import dist
from collections import defaultdict, OrderedDict

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






