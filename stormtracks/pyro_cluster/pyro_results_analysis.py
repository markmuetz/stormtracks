class PyroResultsAnalysis(object):
    def __init__(self):
        scales = [1, 2, 3]
        pressure_levels = [995, 850, 250]
        trackers = ['nearest_neighbour', 'kalman']
        self._tasks = []

        for scale in scales:
            for pressure_level in pressure_levels:
                for tracker_name in trackers:
                    result_key = 'scale:{0};pl:{1};tracker:{2}'.format(scale,
                                                                       pressure_level,
                                                                       tracker_name)
                    self._tasks.append(PyroTask(year, em, result_key))

    def get_next_outstanding(self):
        '''Returns the next outstanding task, None if there are no more'''
        for task in self._tasks:
            if task.status == 'outstanding':
                return task
        return None
