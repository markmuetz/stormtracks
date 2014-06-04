import os
import urllib
import tarfile
import shutil

import requests
from BeautifulSoup import BeautifulSoup as BS

TGZ_DROPBOX_FILES = ['20C_2005_Wilma.tgz', 'TCMIP_algorithm.tgz']

def remove_files(settings):
    pass

def get_data(settings):
    for tgz in TGZ_DROPBOX_FILES: 
        print('extracting file %s'%tgz)
        tar = tarfile.open("/home/markmuetz/Dropbox/UCL MSc/Dissertation/Data/%s"%tgz)
        tar.extractall('data')
        tar.close()

def download_storm_tracks(settings):
    #base_url = 'http://weather.unisys.com/hurricane/atlantic/2005H/index.php'
    base_url = 'http://weather.unisys.com/hurricane/'
    regions = ['atlantic', 'e_pacific', 'w_pacific', 's_pacific', 's_indian', 'n_indian']
    start_year, end_year = 2000, 2014

    if not os.path.exists('data/stormtracks'):
        os.mkdir('data/stormtracks')

    for region in regions:
        print(region)
        if not os.path.exists('data/stormtracks/%s'%region):
            os.mkdir('data/stormtracks/%s'%region)
        for year in range(start_year, end_year):
            y = str(year)
            print('  %s'%y)
            if year <= 2012 and region != 's_pacific':
                h = 'H'
            else:
                h = ''

            url = '%s/%s/%s%s'%(base_url, region, y, h)
            index_html = requests.get('%s/index.php'%url)
            index_bs = BS(index_html.content)
            index_anchors = index_bs.findAll('a')
            for anchor in index_anchors:
                href = anchor['href']
                if (len(href.split('/')) != 0 and
                    href.split('/')[-1] == 'track.dat'):

                    if os.path.exists(('data/stormtracks/%s/%s/%s.dat'%(region, y, href.split('/')[0]))):
                        print('    %s exists'%href)
                        continue

                    print('    %s'%href)
                    track = requests.get('%s/%s'%(url, href))

                    if not os.path.exists('data/stormtracks/%s/%s'%(region, y)):
                        os.mkdir('data/stormtracks/%s/%s'%(region, y))

                    with open('data/stormtracks/%s/%s/%s.dat'%(region, y, href.split('/')[0]), 'w') as f:
                        f.write(track.content)


            

def download_file(url, output_dir, local_name=None):
    if local_name == None:
	local_name = '%s/%s'%(output_dir, url.split('/')[-1])

    print(local_name)
    urllib.urlretrieve(url, local_name)

def download_c20_range(start_year, end_year):
    for year in range(start_year, end_year + 1):
	download_c20(year)

def download_c20(year):
    y = str(year)
    data_dir = 'data/c20/%s'%y
    if not os.path.exists(data_dir):
	os.makedirs(data_dir)

    urls = ['ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2/monolevel/prmsl.%s.nc',
	    'ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2/monolevel/uwnd.sig995.%s.nc',
	    'ftp://ftp.cdc.noaa.gov/Datasets/20thC_ReanV2/monolevel/vwnd.sig995.%s.nc',
	    ]
    print(year)
    for url in urls:
	download_file(url%y, data_dir)

    compress_dir(data_dir)
    print('removing dir %s'%data_dir)
    shutil.rmtree(data_dir)

def compress_dir(data_dir):
    compressed_file = data_dir + '.bz2'
    print('compressing to %s'%compressed_file)
    tar = tarfile.open(compressed_file, 'w:bz2')
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            tar.add(os.path.join(root, file))
    tar.close()

def decompress_file(compressed_file):
    tar = tarfile.open(compressed_file)
    tar.extractall('.')
    tar.close()


def download(settings):
    download_storm_tracks()

if __name__ == "__main__":
    download_storm_tracks(None)

