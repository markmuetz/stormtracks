import tarfile

TGZ_DROPBOX_FILES = ['20C_2005_Wilma.tgz', 'TCMIP_algorithm.tgz']

def remove_files(settings):
    pass

def get_data(settings):
    for tgz in TGZ_DROPBOX_FILES: 
        print('extracting file %s'%tgz)
        tar = tarfile.open("/home/markmuetz/Dropbox/UCL MSc/Dissertation/Data/%s"%tgz)
        tar.extractall('data')
        tar.close()
