#!/usr/bin/env python
import sys
import os
import inspect
import shutil
import subprocess
import pip
from glob import glob

from termcolor import cprint as _cprint
import argh
import argh.helpers

try:
    import stormtracks.setup_logging as setup_logging
    try:
        log = setup_logging.get_logger('st.admin')
    except Exception, e:
        print('Problem creating log file')
        raise
except ImportError:
    print('Problem importing stormtracks.setup_logging')
    raise

CMDS = {'debian': [
    'sudo aptitude install build-essential libhdf5-dev libgeos-dev libproj-dev '\
'libfreetype6-dev python-dev libblas-dev liblapack-dev gfortran libnetcdf-dev',
    'sudo aptitude install python-tk tcl-dev tk-dev',
    'cd /usr/lib/ && sudo ln -s libgeos-3.4.2.so libgeos.so; cd -'],
	'fedora_core': [
    'sudo dnf install gcc-c++ hdf5-devel geos-devel proj-devel blas-devel lapack-devel netcdf-devel freetype-devel',
    'sudo dnf install ScientificPython-tk tcl-devel tk-devel'
    ]
}

PIP_CMDS = (
    'pip install -r requirements/requirements_a.txt',
    'pip install -r requirements/requirements_b.txt',
    'pip install -r requirements/requirements_c.txt',
    'pip install -r requirements/requirements_analysis.txt --allow-external basemap --allow-unverified basemap')


def cprint(text, color=None, on_color=None, attrs=None, **kwargs):
    log.debug(text)
    _cprint(text, color, on_color, attrs, **kwargs)


def log_command(command):
    _cprint(command, 'green', attrs=['bold'])
    log.debug('=' * 80)
    log.debug(command)
    log.debug('=' * 80)
    try:
	result = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
	print(result)
	log.debug('Result:\n{}'.format(result))
    except subprocess.CalledProcessError, e:
	_cprint(e.output, 'red', attrs=['bold'])
	log.warn('Problem with command')
	log.warn(e.output)
	log.warn(e)


def install_os_dependencies(os_type='debian'):
    cprint('Installing OS ({}) requirements'.format(os_type), 'green')
    # These commands may require user input, do directly using call.
    commands = CMDS[os_type]
    for command in commands:
	subprocess.call(command, shell=True)


def copy_files():
    cprint('Copying files to local directory', 'green')
    target_dir = os.getcwd()

    try:
        import stormtracks.installation
        source_dir = os.path.dirname(stormtracks.installation.__file__)
    except ImportError:
        cprint('COULD NOT IMPORT stormtracks.installation', 'red', attrs=['bold'])
        raise

    cprint('Copying from:\n    {}\n  to\n    {}'.format(source_dir, target_dir), 'green')

    if os.path.exists(os.path.join(target_dir, 'requirements')):
	shutil.rmtree('requirements', ignore_errors=True)
    shutil.copytree(os.path.join(source_dir, 'requirements'), os.path.join(target_dir, 'requirements'))

    if os.path.exists(os.path.join(target_dir, 'stormtracks_settings.py')):
        # os.rename(os.path.join(target_dir, 'stormtracks_settings.py'), os.path.join(target_dir, 'stormtracks_settings.py.bak'))
        cprint('stormtracks_settings.py already exists, skipping', 'yellow')
    else:
        shutil.copyfile(os.path.join(source_dir, 'settings', 'default_stormtracks_settings.py'),
                        os.path.join(target_dir, 'stormtracks_settings.py'))

    # shutil.copytree(os.path.join(source_dir, 'classifiers'), os.path.join(target_dir, 'classifiers'))
    # shutil.copytree(os.path.join(source_dir, 'plots'), os.path.join(target_dir, 'plots'))
    for script_file in glob(os.path.join(source_dir, 'scripts/*.py')):
        if os.path.basename(script_file) == '__init__.py':
            continue
        if os.path.exists(os.path.join(target_dir, os.path.basename(script_file))):
            cprint('{} already exists, skipping'.format(os.path.basename(script_file), 'yellow'))
        else:
            shutil.copyfile(script_file, os.path.basename(script_file))


def install(use_log=True):
    cprint('Installing all dependencies', 'green', attrs=['bold'])
    log_info()
    copy_files()
    install_pip()
    log_info()


def install_pip():
    cprint('Installing pip requirements', 'green')

    for command in PIP_CMDS:
	subprocess.call(command, shell=True)


def print_installation_commands(os_type='debian'):
    print('{} copy-files'.format(os.path.basename(sys.argv[0])))
    commands = CMDS[os_type]
    for command in commands:
	print(command)
    for command in PIP_CMDS:
        print(command)


def list_data_sources(c20version='v1', full=False):
    from stormtracks.load_settings import settings
    from stormtracks.c20data import C20_DATA_DIR
    from stormtracks.ibtracsdata import IBTRACS_DATA_DIR
    cprint('Using data dir: {}'.format(settings.DATA_DIR), 'green', attrs=['bold'])

    cprint('20CR Data Sources ({}):'.format(C20_DATA_DIR), 'green')
    for c20year in glob(os.path.join(C20_DATA_DIR, c20version, '*')):
        cprint('  {}'.format(c20year), 'blue', attrs=['bold'])
        for c20file in sorted(glob(os.path.join(c20year, '*.nc'))):
            cprint('    {}'.format(c20file), 'blue')

    ibtracs_tarball_file = os.path.join(settings.DATA_DIR, 'ibtracs_v03r05_dataset_184210_201305.tar.gz')

    cprint('IBTrACS Data Sources ({}):'.format(IBTRACS_DATA_DIR), 'green')
    cprint('  IBTrACS tarball downloaded: {}'.format(os.path.exists(ibtracs_tarball_file)), 'blue', attrs=['bold'])
    cprint('  IBTrACS tarball decompressed: {}'.format(os.path.exists(settings.IBTRACS_DATA_DIR)), 'blue', attrs=['bold'])
    if full:
        for ibfile in glob(os.path.join(settings.IBTRACS_DATA_DIR, '*')):
            cprint('  {}'.format(ibfile), 'blue')


def list_output(full=False):
    from stormtracks.load_settings import settings
    from stormtracks.results import StormtracksResultsManager
    cprint('Using output dir: {}'.format(settings.OUTPUT_DIR), 'green', attrs=['bold'])
    cprint('  Tracking results: {}'.format(settings.RESULTS), 'green', attrs=['bold'])
    results_manager = StormtracksResultsManager(settings.RESULTS)
    results_years = results_manager.list_years()
    for year in results_years:
        results = results_manager.list_results(year)
        for result_name in results:
            cprint('    Tracking results year: {}; {}'.
                   format(year, result_name), 'blue', attrs=['bold'])


def install_full(os_type='debian'):
    log_info()
    install_os_dependencies(os_type)
    install(False)
    log_info()


def log_info():
    log.debug('Running from: {}'.format(__file__))
    log.debug('in virtualenv: {}'.format(hasattr(sys, 'real_prefix')))
    commands = (
	'uname -a',
	'cat /etc/lsb-release',
	'dpkg -l|grep ^ii|awk \'{print $2 "\t" $3}\'',
	'pip freeze',
	'git rev-parse HEAD',
	)

    for command in commands:
	log_command(command)


def version():
    import stormtracks
    print(stormtracks.__version__)


def main():
    log.debug(' '.join(sys.argv))
    parser = argh.helpers.ArghParser()
    argh.add_commands(parser, [print_installation_commands, install, install_full, 
                               list_data_sources, list_output, log_info, copy_files, 
                               version])
    argh.dispatch(parser)


if __name__ == '__main__':
    main()
