import os

# Directories where data and output will be saved.
SETTINGS_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.expandvars('$HOME/stormtracks_data/data')
OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/output')
SECOND_OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/output')
LOGGING_DIR = 'logs'
FIGURE_OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/figures')

# 20th C Reanalysis project version.
C20_VERSION = 'v2'

# Lat/Lon range.
MIN_LON = 260
MAX_LON = 340
MIN_LAT = 0
MAX_LAT = 60

RESULTS = 'prod_release_1'

CONSOLE_LOG_LEVEL = 'info'
FILE_LOG_LEVEL = 'debug'

CHUNK_SIZE = 1024*1000
MINIMUM_DOWNLOAD_RATE_1 = 300000 # B/s - 0.3 MB/s.
MINIMUM_DOWNLOAD_RATE_2 = 1000000 # B/s - 1 MB/s.
