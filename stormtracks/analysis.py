import datetime as dt

from results import StormtracksResultsManager
from ibtracsdata import IbtracsData
from c20data import C20Data, GlobalEnsembleMember
from tracking import VortmaxFinder, VortmaxNearestNeighbourTracker, VortmaxKalmanFilterTracker
from match import match


class StormtracksAnalysis(object):
    def __init__(self, year, ensemble_member):
        self.year = year
        self.ensemble_member = ensemble_member

        self.ibdata = IbtracsData(verbose=False)
        self.best_tracks = self.ibdata.load_ibtracks_year(year)
        self.results = {}

        self.results_manager = StormtracksResultsManager('analysis')

    def run_analysis(self):
        scales = [1, 2, 3]
        pressure_levels = [995, 850, 250]
        trackers = ['nearest_neighbour', 'kalman']

        for scale in scales:
            for pressure_level in pressure_levels:
                for tracker_name in trackers:
                    result_key = 'scale:{0};pl:{1};tracker:{2}'.format(scale,
                                                                       pressure_level,
                                                                       tracker_name)
                    saved_results = self.results_manager.list_results(self.year,
                                                                      self.ensemble_member)
                    if result_key not in saved_results:
                        print('Running analysis: {0}'.format(result_key))
                        result = self.run_individual_analysis(scale, pressure_level, tracker_name)
                        self.results_manager.add_result(self.year,
                                                        self.ensemble_member,
                                                        result_key,
                                                        result)
                        self.results_manager.save()
                    else:
                        print('Loading saved result: {0}'.format(result_key))
                        self.results_manager.load(self.year,
                                                  self.ensemble_member,
                                                  result_key)

    def run_individual_analysis_from_results_key(self, key):
        config = key.split(';')
        scale = int(config[0].split(':')[1])
        pressure_level = int(config[1].split(':')[1])
        tracker_name = config[2].split(':')[1]
        result = self.run_individual_analysis(scale, pressure_level, tracker_name)
        return result

    def run_individual_analysis(self, scale, pressure_level, tracker_name):
        print('Scale: {0}; Press level: {1}; Tracker: {2}'.format(
            scale, pressure_level, tracker_name))

        if scale == 1:
            upscaling = False
        else:
            upscaling = True
            
        c20data = C20Data(self.year, verbose=False, 
                          pressure_level=pressure_level,
                          upscaling=upscaling, 
                          scale_factor=scale)

        if tracker_name == 'nearest_negighbour':
            tracker = VortmaxNearestNeighbourTracker()
        elif tracker_name == 'kalman':
            tracker = VortmaxKalmanFilterTracker()

        gem = GlobalEnsembleMember(c20data, self.ensemble_member)
        vort_finder = VortmaxFinder(gem)

        vort_finder.find_vort_maxima(dt.datetime(self.year, 6, 1), 
                                     dt.datetime(self.year, 12, 1),
                                     use_upscaled=upscaling)

        tracker.track_vort_maxima(vort_finder.vortmax_time_series)

        matches = match(tracker.vort_tracks_by_date, self.best_tracks)
        good_matches = [ma for ma in matches.values() if ma.av_dist() < 5 and ma.overlap > 6]

        return good_matches
