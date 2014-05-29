import os

import tarfile
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


            



def download(settings):
    download_storm_tracks()

if __name__ == "__main__":
    download_storm_tracks(None)

