import argh

import stormtracks.download as download
from stormtracks.setup_logging import get_logger

log = get_logger('st.demo')


def download_year(year=2005):
    '''Download (all) IBTrACS and 20CR data for given year.
    
    Can safely be run repeatedly without redownloading any unneccessary data.
    Will detect any partially downloaded files and delete/redownload them.'''
    log.info('Downloading year {}'.format(year))

    # By default data will be saved to ~/stormtracks_data/data/
    download.download_ibtracs()
    # N.B. one year is ~12GB of data! This will take a while.
    download.download_full_c20(year)


if __name__ == '__main__':
    argh.dispatch_command(download_year)
