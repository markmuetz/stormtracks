# *** DON'T MODIFY THIS FILE! ***
#
# Instead copy it to stormtracks_settings.py
#
# Default settings for project
# This will get copied to $HOME/.stormtracks/
# on install.
import os

SETTINGS_DIR = os.path.abspath(os.path.dirname(__file__))
# expandvars expands e.g. $HOME
DATA_DIR = os.path.expandvars('$HOME/stormtracks_data/data')
TMP_DATA_DIR = None
OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/output')
LOGGING_DIR = os.path.expandvars('$HOME/stormtracks_data/logs')
FIGURE_OUTPUT_DIR = os.path.expandvars('$HOME/stormtracks_data/figures')

C20_FULL_DATA_DIR = os.path.join(DATA_DIR, 'c20_full')
C20_GRIB_DATA_DIR = os.path.join(DATA_DIR, 'c20_grib')
C20_MEAN_DATA_DIR = os.path.join(DATA_DIR, 'c20_mean')
IBTRACS_DATA_DIR = os.path.join(DATA_DIR, 'ibtracs')
