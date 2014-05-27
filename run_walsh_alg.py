import subprocess
import os, io
from glob import glob
from argparse import ArgumentParser

from jinja2 import Environment, FileSystemLoader

def run(args):
    symlinks = []
    if args.walsh:
        fname = 'example'
        y = '1999'
	#months = ('01', '02')
	months = ('01',)
        data_dir = 'data/TCMIP_algorithm/example_tclv'
	tpl_file = 'nml_default.tpl'
    elif args.c20:
        fname = 'c20'
        y = '2005'
        data_dir = 'processed_data/'
	months = ('08',)
	tpl_file = 'nml_c20default.tpl'


    for nc_file in glob('%s/%s_%s*.nc'%(data_dir, fname, y)):
        symlink = 'bin/%s'%os.path.basename(nc_file)
        symlinks.append(symlink)
        if not os.path.exists(symlink):
            os.symlink(os.path.abspath(nc_file), symlink)

    for m in months:
        env = Environment(loader=FileSystemLoader('settings'))
        tpl = env.get_template(tpl_file)
        with io.open('bin/nml.nml', 'w') as f:
            f.write(tpl.render(fname="'%s_%s%s.nc'"%(fname, y, m), prefix="'tclv_out'"))
            f.write(u'\n')

        os.chdir('bin')
        if args.modified:
            subprocess.call('./cyclone_modified', shell=True)
        else:
            subprocess.call('./cyclone', shell=True)
        os.chdir('../')

    for symlink in symlinks:
        os.remove(symlink)
        pass

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-m', '--modified', help='Use modified cyclone exe', action='store_true')
    parser.add_argument('-w', '--walsh', help='Use Walsh example data', action='store_true')
    parser.add_argument('-c', '--c20', help='Use C20 data', action='store_true')
    args = parser.parse_args()
    run(args)


