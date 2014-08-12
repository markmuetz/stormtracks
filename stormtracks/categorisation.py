from collections import OrderedDict


class CutoffCategoriser(object):
    def __init__(self):
        self.vort_cutoff = 0.00035
        self.pmin_cutoff = 99000
        self.min_dist_cutoff = 400
        self.t995_cutoff = 295
        self.t850_cutoff = 287
        self.t_anom_cutoff = -4
    
    def categorise(self, cyclone):
        cyclone.hurricane_cat = OrderedDict()
        for date in cyclone.dates:
            if cyclone.vortmax_track.vortmax_by_date[date].pos[0] < 260:
                cyclone.hurricane_cat[date] = False
            elif cyclone.vortmax_track.vortmax_by_date[date].vort < self.vort_cutoff:
                cyclone.hurricane_cat[date] = False
            elif cyclone.pmins[date] < self.pmin_cutoff:
                cyclone.hurricane_cat[date] = False
            elif cyclone.min_dists[date] > self.min_dist_cutoff:
                cyclone.hurricane_cat[date] = False
            elif cyclone.t995s[date] < self.t995_cutoff:
                cyclone.hurricane_cat[date] = False
            elif cyclone.t850s[date] < self.t850_cutoff:
                cyclone.hurricane_cat[date] = False
            elif cyclone.t850s[date] - cyclone.t995s[date] > self.t_anom_cutoff:
                cyclone.hurricane_cat[date] = False
            else:
                cyclone.hurricane_cat[date] = True
