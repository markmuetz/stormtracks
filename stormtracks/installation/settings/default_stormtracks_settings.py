import os

SETTINGS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.expandvars('$HOME/stormtracks_data/data')
OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/output')
SECOND_OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/output')
LOGGING_DIR = os.path.expandvars('$HOME/stormtracks_data/logs')
FIGURE_OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/figures')

C20_FULL_DATA_DIR = os.path.join(DATA_DIR, 'c20_full')
C20_GRIB_DATA_DIR = os.path.join(DATA_DIR, 'c20_grib')
C20_MEAN_DATA_DIR = os.path.join(DATA_DIR, 'c20_mean')
IBTRACS_DATA_DIR = os.path.join(DATA_DIR, 'ibtracs')

TRACKING_RESULTS = 'prod_release_1'
FIELD_RESULTS = 'prod_release_1'
