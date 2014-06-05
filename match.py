from detect import dist

def match(cyclones_by_date, tracks):
    matched = []
    matches = {}
    for track in tracks:
	for lon, lat, date in zip(track.lon, track.lat, track.dates):
	    if date in cyclones_by_date.keys():
		cyclones = cyclones_by_date[date]
		for c in cyclones:
		    print(c.cell_pos)
		    print(lon + 360, lat)
		    if dist(c.cell_pos, (lon + 180, lat)) < 10:
			matches[track] = c
    return matches






