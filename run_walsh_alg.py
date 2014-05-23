import subprocess
import os, io
from glob import glob

from jinja2 import Environment, FileSystemLoader

def run():
    symlinks = []
    if False:
        fname = 'example'
        y = '1999'
	months = ('01', '02')
        data_dir = 'data/TCMIP_algorithm/example_tclv'
    else:
        fname = 'c20'
        y = '2005'
        data_dir = 'processed_data/'
	months = ('10',)


    for nc_file in glob('%s/%s_%s*.nc'%(data_dir, fname, y)):
        symlink = 'bin/%s'%os.path.basename(nc_file)
        symlinks.append(symlink)
        if not os.path.exists(symlink):
            os.symlink(os.path.abspath(nc_file), symlink)

    for m in months:
        env = Environment(loader=FileSystemLoader('settings'))
        tpl = env.get_template('nml.default.tpl')
        with io.open('bin/nml.nml', 'w') as f:
            f.write(tpl.render(fname="'%s_%s%s.nc'"%(fname, y, m), prefix="'tclv_out'"))
            f.write(tpl.render(fname="'%s_%s%s.nc'"%(fname, y, m), prefix="'tclv_out'"))
            f.write(u'\n')

        os.chdir('bin')
        subprocess.call('./cyclone', shell=True)
        os.chdir('../')

    for symlink in symlinks:
        os.remove(symlink)

if __name__ == "__main__":
    run()


