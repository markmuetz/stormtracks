import stormtracks.download as dl

# By default data will be saved to ~/stormtracks/data/
dl.download_ibtracs()
# N.B. one year is ~12GB of data! This will take a while.
dl.download_full_c20(2005)
