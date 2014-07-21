import os
import urllib
import tarfile
import shutil

from logger import Logger
from load_settings import settings

C20_FULL_DATA_DIR = settings.C20_FULL_DATA_DIR
C20_MEAN_DATA_DIR = settings.C20_MEAN_DATA_DIR
DATA_DIR = settings.DATA_DIR
TMP_DATA_DIR = settings.TMP_DATA_DIR

log = Logger('download', 'download.log').get()


def _download_file(url, output_dir, tmp_output_dir=None, path=None):
    if path is None:
        path = os.path.join(output_dir, url.split('/')[-1])

    if tmp_output_dir:
        tmp_path = os.path.join(tmp_output_dir, url.split('/')[-1])
    else:
        tmp_path = None

    if os.path.exists(path):
        log.info('Already exists, skipping')
    else:
        if tmp_path:
            log.info(tmp_path)
            urllib.urlretrieve(url, path)
        else:
            log.info(path)
            urllib.urlretrieve(url, tmp_path)

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
    _decompress_file(path)


def download_mean_c20_range(start_year, end_year):
    '''Downloads mean values for prmsl, u and v in a given range'''
    for year in range(start_year, end_year + 1):
        download_mean_c20(year)


def download_full_c20_range(start_year, end_year):
    '''Downloads each ensemble member's values for prmsl, u and v in a given range'''
    for year in range(start_year, end_year + 1):
        download_full_c20(year)


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

    _compress_dir(data_dir)
    log.info('removing dir {0}'.format(data_dir))
    shutil.rmtree(data_dir)


def download_full_c20(year):
    '''Downloads each ensemble member's values for prmsl, u and v'''
    y = str(year)
    data_dir = os.path.join(C20_FULL_DATA_DIR, y)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if TMP_DATA_DIR:
        tmp_data_dir = os.path.join(TMP_DATA_DIR, y)
        if not os.path.exists(tmp_data_dir):
            os.makedirs(tmp_data_dir)
    else:
        tmp_data_dir = data_dir

    # urls = ['http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/u9950/u9950_{0}.nc',
    #         'http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/v9950/v9950_{0}.nc',
    #         'http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/prmsl/prmsl_{0}.nc',
    urls = ['http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/u850/u850_{0}.nc',
            'http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/v850/v850_{0}.nc',
            'http://portal.nersc.gov/pydap/20C_Reanalysis_ensemble/analysis/prmsl/prmsl_{0}.nc',
            ]
    log.info(year)
    for url in urls:
        _download_file(url.format(year), data_dir, tmp_data_dir)

    if TMP_DATA_DIR:
        shutil.copytree(tmp_data_dir, data_dir)
        shutil.rmtree(tmp_data_dir)

    # These files are incompressible (already compressed I guess)
    # Hence no need to call e.g.:
    # _compress_dir(data_dir)


def _compress_dir(data_dir):
    compressed_file = data_dir + '.bz2'
    log.info('compressing to {0}'.format(compressed_file))
    tar = tarfile.open(compressed_file, 'w:bz2')
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            tar.add(os.path.join(root, file))
    tar.close()


def _decompress_file(compressed_file):
    log.info('decompressing {0}'.format(compressed_file))
    tar = tarfile.open(compressed_file)
    tar.extractall(os.path.dirname(compressed_file))
    tar.close()


if __name__ == "__main__":
    download_ibtracs()
    # Will take a while, each year is 4.2GB of data.
    download_full_c20(2005)
