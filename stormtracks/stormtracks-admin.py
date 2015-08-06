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


def install_aptitude():
    cprint('Installing OS (Debian/Ubuntu) requirements', 'green')
    log_command('sudo aptitude install build-essential libhdf5-dev libgeos-dev libproj-dev libfreetype6-dev python-dev libblas-dev liblapack-dev gfortran libnetcdf-dev python-tk tcl-dev tk-dev')
    log_command('cd /usr/lib/ && sudo ln -s libgeos-3.4.2.so libgeos.so')


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


def install():
    cprint('Installing all dependencies', 'green', attrs=['bold'])
    copy_files()
    install_pip()


def install_pip():
    cprint('Installing pip requirements', 'green')

    pip_commands = (
	'pip install -r requirements/requirements_a.txt',
	'pip install -r requirements/requirements_b.txt',
	'pip install -r requirements/requirements_c.txt',
	'pip install -r requirements/requirements_analysis.txt --allow-external basemap --allow-unverified basemap')

    for command in pip_commands:
	log_command(command)



def clean():
    shutil.rmtree('requirements', ignore_errors=True)
    shutil.rmtree('classifiers', ignore_errors=True)
    shutil.rmtree('plots', ignore_errors=True)
    try:
        os.remove('stormtracks_settings.py')
    except OSError:
        pass


def list_data_sources(c20version='v1', full=False):
    from stormtracks.load_settings import settings
    cprint('Using data dir: {}'.format(settings.DATA_DIR), 'green', attrs=['bold'])

    cprint('20CR Data Sources ({}):'.format(settings.C20_FULL_DATA_DIR), 'green')
    for c20year in glob(os.path.join(settings.C20_FULL_DATA_DIR, c20version, '*')):
        cprint('  {}'.format(c20year), 'blue', attrs=['bold'])
        for c20file in sorted(glob(os.path.join(c20year, '*.nc'))):
            cprint('    {}'.format(c20file), 'blue')

    ibtracs_tarball_file = os.path.join(settings.DATA_DIR, 'ibtracs_v03r05_dataset_184210_201305.tar.gz')

    cprint('IBTrACS Data Sources ({}):'.format(settings.IBTRACS_DATA_DIR), 'green')
    cprint('  IBTrACS tarball downloaded: {}'.format(os.path.exists(ibtracs_tarball_file)), 'blue', attrs=['bold'])
    cprint('  IBTrACS tarball decompressed: {}'.format(os.path.exists(settings.IBTRACS_DATA_DIR)), 'blue', attrs=['bold'])
    if full:
        for ibfile in glob(os.path.join(settings.IBTRACS_DATA_DIR, '*')):
            cprint('  {}'.format(ibfile), 'blue')


def list_output(full=False):
    from stormtracks.load_settings import settings
    from stormtracks.results import StormtracksResultsManager
    cprint('Using output dir: {}'.format(settings.OUTPUT_DIR), 'green', attrs=['bold'])
    cprint('  Tracking results: {}'.format(settings.TRACKING_RESULTS), 'green', attrs=['bold'])
    results_manager = StormtracksResultsManager(settings.TRACKING_RESULTS)
    results_years = results_manager.list_years()
    for year in results_years:
        results = results_manager.list_results(year)
        for result_name in results:
            cprint('    Tracking results year: {}; {}'.
                   format(year, result_name), 'blue', attrs=['bold'])


def install_full():
    install_aptitude()
    install()


def reinstall():
    clean()
    install()


def log_info():
    log.debug('from: {}'.format(__file__))
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
    


def main():
    log.debug(' '.join(sys.argv))
    parser = argh.helpers.ArghParser()
    argh.add_commands(parser, [install, install_full, clean, reinstall, list_data_sources,
        list_output, log_info])

    argh.dispatch(parser)


if __name__ == '__main__':
    main()
