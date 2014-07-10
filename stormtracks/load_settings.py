import os, sys

# appends user's stormtracks dir to the path.
settings_dir = os.path.join(os.path.expandvars('$HOME'), '.stormtracks')
sys.path.append(settings_dir)

import stormtracks_settings as settings
