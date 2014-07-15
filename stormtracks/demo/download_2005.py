import stormtracks.download as dl

# Data will be saved to ~/stormtracks/data/
dl.download_ibtracs()
# N.B. one year is 4.2 GB of data! This will take a while.
# It will download 3 files, two with the wind velocities at ~sea level (u9950/v9950)
# and Pressure at Sea Level (prmsl).
dl.download_full_c20(2005)
