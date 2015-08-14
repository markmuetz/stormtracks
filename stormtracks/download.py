import os
import urllib
import shutil
from glob import glob
import datetime as dt
import hashlib

import requests

import setup_logging
from load_settings import settings
from utils.utils import compress_dir, decompress_file

C20_FULL_DATA_DIR = os.path.join(settings.DATA_DIR, 'c20_full')
C20_GRIB_DATA_DIR = os.path.join(settings.DATA_DIR, 'c20_grib')
C20_MEAN_DATA_DIR = os.path.join(settings.DATA_DIR, 'c20_mean')

DATA_DIR = settings.DATA_DIR

CHUNK_SIZE = settings.CHUNK_SIZE
MINIMUM_DOWNLOAD_RATE_1 = settings.MINIMUM_DOWNLOAD_RATE_1
MINIMUM_DOWNLOAD_RATE_2 = settings.MINIMUM_DOWNLOAD_RATE_2

log = setup_logging.get_logger('st.download')


def _download_file(url, output_dir):
    log.info('Downloading url: {}'.format(url))
    path = os.path.join(output_dir, url.split('/')[-1])
    sha1path = path + '.sha1sum'
    log.info('Download to : {}'.format(path))

    if url[:3].lower() == 'ftp':
        path = _ftp_download_file(url, path)
    else:
        path = _min_download_speed_download_file(url, path)

    if path is not None:
        log.info('Downloaded')
        with open(sha1path, 'w') as sha1_file:
            sha1 = sha1_of_file(path)
            sha1_file.write(sha1)
            log.info('Calculated sha1sum: {}'.format(sha1))
    return path


def _ftp_download_file(url, path):
    if os.path.exists(path):
        info = urllib.urlopen(url)
        total_length = long(info.headers.get('content-length'))

        if os.stat(path).st_size != total_length:
            log.info('File already exists, but is incomplete, deleting')
            os.remove(path)
        else:
            log.info('File already exists, skipping')
            log.info('path: {0}'.format(path))
            return None

    urllib.urlretrieve(url, path)
    return path


def _min_download_speed_download_file(url, path):
    log.info('Min. download speeds: {}, {}'.format(MINIMUM_DOWNLOAD_RATE_1, MINIMUM_DOWNLOAD_RATE_2))

    response = requests.get(url, stream=True)
    total_length = long(response.headers.get('content-length'))

    if os.path.exists(path):
        if os.stat(path).st_size != total_length:
            log.info('File already exists, but is incomplete, deleting')
            os.remove(path)
        else:
            log.info('File already exists, skipping')
            log.info('path: {0}'.format(path))
            return None

    while True:
        redownload = False
        start_time = dt.datetime.now()
        total_downloaded = 0

        try:
            with open(path, 'w') as dl_file:
                for data in response.iter_content(CHUNK_SIZE):
                    download_ratio = 1. * total_downloaded / total_length
                    total_downloaded += len(data)
                    dl_file.write(data)
                    elapsed_seconds = (dt.datetime.now() - start_time).total_seconds()
                    log.debug('Avg. dl speed: {0:6.2f} MB/s, {1:5.2f}s, {2:3.1f}%'
                              .format(total_downloaded / (1e6 * elapsed_seconds), 
                                      elapsed_seconds, 
                                      100 * download_ratio))
                    if ((elapsed_seconds > 20 and total_downloaded / elapsed_seconds < MINIMUM_DOWNLOAD_RATE_1) or 
                        (elapsed_seconds > 60 and total_downloaded / elapsed_seconds < MINIMUM_DOWNLOAD_RATE_2)):
                        redownload = True
                        break
        except requests.exceptions.Timeout:
            redownload = True

        if redownload:
            os.remove(path)
            message_tpl = 'Download speed: {0:6.2f} MB/s, after {1:5.2f}s. Redownloading'
            elapsed_seconds = (dt.datetime.now() - start_time).total_seconds()
            log.info(message_tpl.format(total_downloaded / (1e6 * elapsed_seconds), elapsed_seconds))

            response = requests.get(url, stream=True)
        else:
            break
    return path


def sha1_of_file(filepath):
    with open(filepath, 'rb') as f:
        sha1 = hashlib.sha1()
        while True:
            buf = f.read(0x100000)
            if not buf:
                break
            sha1.update(buf)

        return sha1.hexdigest()


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
    if path is not None:
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


def download_full_c20(year, variables='all', version='v1'):
    '''Downloads each ensemble member's values for given variables'''
    y = str(year)
    data_dir = os.path.join(C20_FULL_DATA_DIR, version, y)
    log.info('Using data dir {0}'.format(data_dir))

    if version == 'v1':
        # Old version of 20CR:
        url_tpl = 'http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/{var}/{var}_{year}.nc'
    elif version == 'v2':
        url_tpl = 'http://portal.nersc.gov/pydap/20C_Reanalysis_version2c_ensemble/analysis/{var}/{var}_{year}.nc'
    else:
        Exception('Unrecognized version: {}'.format(version))

    if variables == 'all':
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
            # 'rh9950', # No longer get this by default.
            'pwat']

    log.info('Downloading vars: {}'.format(', '.join(variables)))

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    log.info('Downloading year {0}'.format(year))
    for variable in variables:
        url = url_tpl.format(year=year, var=variable)
        log.info('url: {0}'.format(url))
        _download_file(url, data_dir)
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
