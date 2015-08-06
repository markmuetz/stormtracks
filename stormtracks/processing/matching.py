import datetime as dt

import pandas as pd

from .. import setup_logging
from ..utils.utils import geo_dist

log = setup_logging.get_logger('st.matching')


def simple_matching(best_tracks, df):
    start = dt.datetime.now()
    print(start)

    bt_matches = []
    index = []
    for bt in best_tracks:
        for bt_index, date in enumerate(bt.dates):
            print(date)
            curr_inds = df.index[df.date == date]

            min_dists = [{'bt_name': bt.name, 'bt_min_dist': 1e99, 'index': None} for i in range(56)]
            for i in curr_inds:
                row = df.iloc[i]
                vortmax_pos = (row.lon, row.lat)
                min_dist = min_dists[row.em]['bt_min_dist']
                bt_pos = (bt.lons[bt_index], bt.lats[bt_index])
                dist = geo_dist(vortmax_pos, bt_pos)
                if dist < min_dist:
                    min_dists[row.em]['bt_min_dist'] = dist
                    min_dists[row.em]['index'] = i
                    min_dists[row.em]['bt_wind'] = bt.winds[bt_index]
                    min_dists[row.em]['is_hurr'] = bt.cls[bt_index] == 'HU'

            for min_dist in min_dists:
                if min_dist['index'] is not None:
                    bt_matches.append(min_dist)
                    index.append(min_dist['index'])

    end = dt.datetime.now()
    print(end - start)
    return pd.DataFrame(bt_matches, index=index, columns=['bt_name', 'bt_min_dist', 'bt_wind', 'is_hurr'])
