import os
import urllib
import shutil
from glob import glob

from logger import setup_logging
from load_settings import settings
from utils.utils import compress_dir, decompress_file

C20_FULL_DATA_DIR = settings.C20_FULL_DATA_DIR
C20_GRIB_DATA_DIR = settings.C20_GRIB_DATA_DIR
C20_MEAN_DATA_DIR = settings.C20_MEAN_DATA_DIR
DATA_DIR = settings.DATA_DIR

log = setup_logging('download', 'download.log')


def _download_file(url, output_dir, path=None):
    if path is None:
        path = os.path.join(output_dir, url.split('/')[-1])

    if os.path.exists(path):
        log.info('File already exists, skipping')
    else:
        log.info(path)
        urllib.urlretrieve(url, path)

    return path


def download_ibtracs():
    '''Downloads all IBTrACS data

    Downloads compressed tarball from FTP site to settings.DATA_DIR.
    Decompresses it to settings.DATA_DIR/ibtracs
    '''
    url = ('ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v03r05/archive/'
           'ibtracs_v03r05_dataset_184210_201305.tar.gz')
    data_dir = DATA_DIR
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # tarball will be downloaded to data_dir.
    path = _download_file(url, data_dir)
    # it will be decompressed to data_dir/ibtracs
    decompress_file(path)


def download_mean_c20_range(start_year, end_year):
    '''Downloads mean values for prmsl, u and v in a given range'''
    for year in range(start_year, end_year + 1):
        download_mean_c20(year)


def download_full_c20_range(start_year, end_year, variables=None):
    '''Downloads each ensemble member's values for prmsl, u and v in a given range'''
    for year in range(start_year, end_year + 1):
        download_full_c20(year, variables)


def download_mean_c20(year):
    '''Downloads mean values for prmsl, u and v'''
    y = str(year)
    data_dir_tpl = os.path.join(C20_MEAN_DATA_DIR, y)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    urls = ['ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2/monolevel/prmsl.{0}.nc',
            'ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2/monolevel/uwnd.sig995.{0}.nc',
            'ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2/monolevel/vwnd.sig995.{0}.nc',
            ]
    log.info(year)
    for url in urls:
        _download_file(url.format(year), data_dir)

    compress_dir(data_dir)
    log.info('removing dir {0}'.format(data_dir))
    shutil.rmtree(data_dir)


def download_full_c20(year, variables=None):
    '''Downloads each ensemble member's values for prmsl, u and v'''
    y = str(year)
    data_dir = os.path.join(C20_FULL_DATA_DIR, y)
    log.info('Using data dir {0}'.format(data_dir))

    if not variables:
        variables = [
            'prmsl',
            'u850',
            'u9950',
            'v9950',
            'v850',
            # 'u250',
            # 'v250',
            # 't2m', doesn't exist?
            't9950',
            't850',
            'cape',
            # 'rh9950',
            'pwat']

    data_dir = data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url_tpl = 'http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/{var}/{var}_{year}.nc'

    log.info('Downloading year {0}'.format(year))
    for variable in variables:
        url = url_tpl.format(year=year, var=variable)
        log.info('url: {0}'.format(url))
        _download_file(url, data_dir, data_dir)
        log.info('Downloaded')

    # These files are incompressible (already compressed I guess)
    # Hence no need to call e.g.:
    # compress_dir(data_dir)


def delete_full_c20(year):
    '''Deletes all data for given year.'''
    y = str(year)
    data_dir = os.path.join(C20_FULL_DATA_DIR, y)
    log.info('Deleting data dir {0}'.format(data_dir))
    shutil.rmtree(data_dir)


def download_grib_c20(year=2005, month=10, ensemble_member=56):
    '''Downloads the raw data for one ensemble member.

    Contains all fields for the given ensemble member (over 100 of them).
    '''
    url_tpl = ('http://portal.nersc.gov/archive/home/projects/incite11/www/'
               '20C_Reanalysis/everymember_full_analysis_fields/{0}/{0}{1}_pgrbanl_mem{2}.tar')

    data_dir = os.path.join(C20_GRIB_DATA_DIR, str(year))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    downloaded_file = \
        _download_file(url_tpl.format(year, month, ensemble_member), data_dir)
    decompress_file(downloaded_file)


if __name__ == "__main__":
    download_ibtracs()
    # Will take a while, each year is 4.2GB of data.
    download_full_c20(2005)
