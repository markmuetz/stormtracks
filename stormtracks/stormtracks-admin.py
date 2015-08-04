#!/usr/bin/env python
import sys
import os
import inspect
import shutil
import subprocess
import pip
from glob import glob

from termcolor import cprint


def install_aptitude():
    cprint('Installing OS (Debian/Ubuntu) requirements', 'green')
    subprocess.call('sudo aptitude install build-essential libhdf5-dev libgeos-dev libproj-dev libfreetype6-dev python-dev libblas-dev liblapack-dev gfortran libnetcdf-dev python-tk tcl-dev tk-dev', shell=True)
    subprocess.call('cd /usr/lib/ && sudo ln -s libgeos-3.4.2.so libgeos.so', shell=True)


def copy_files():
    cprint('Copying files to local directory', 'green')
    target_dir = os.getcwd()

    try:
        import stormtracks.installation
        source_dir = os.path.dirname(stormtracks.installation.__file__)
    except ImportError:
        cprint('COULD NOT IMPORT stormtracks.installation', 'red')
        raise

    cprint('Copying from:\n    {}\n  to\n    {}'.format(source_dir, target_dir), 'green')
    shutil.copytree(os.path.join(source_dir, 'requirements'), os.path.join(target_dir, 'requirements'))

    shutil.copyfile(os.path.join(source_dir, 'settings', 'default_stormtracks_settings.py'),
                    os.path.join(target_dir, 'default_stormtracks_settings.py'))
    if os.path.exists(os.path.join(target_dir, 'stormtracks_settings.py')):
        # os.rename(os.path.join(target_dir, 'stormtracks_settings.py'), os.path.join(target_dir, 'stormtracks_settings.py.bak'))
        cprint('stormtracks_settings.py already exists, skipping', 'yellow')
    else:
        shutil.copyfile(os.path.join(source_dir, 'settings', 'default_stormtracks_settings.py'),
                        os.path.join(target_dir, 'stormtracks_settings.py'))

    shutil.copytree(os.path.join(source_dir, 'classifiers'), os.path.join(target_dir, 'classifiers'))
    shutil.copytree(os.path.join(source_dir, 'plots'), os.path.join(target_dir, 'plots'))
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

    pip.main(['install', '-r', 'requirements/requirements_a.txt'])
    pip.main(['install', '-r', 'requirements/requirements_b.txt'])
    pip.main(['install', '-r', 'requirements/requirements_c.txt'])
    pip.main(['install', '-r', 'requirements/requirements_analysis.txt', '--allow-external', 'basemap', '--allow-unverified', 'basemap'])


def clean():
    shutil.rmtree('requirements', ignore_errors=True)
    shutil.rmtree('classifiers', ignore_errors=True)
    shutil.rmtree('plots', ignore_errors=True)
    try:
        os.remove('stormtracks_settings.py')
    except OSError:
        pass


def list_data_sources(full=False):
    from stormtracks.load_settings import settings
    cprint('Using data dir: {}'.format(settings.DATA_DIR), 'green', attrs=['bold'])

    cprint('20CR Data Sources ({}):'.format(settings.C20_FULL_DATA_DIR), 'green')
    for c20year in glob(os.path.join(settings.C20_FULL_DATA_DIR, '*')):
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
    from stormtracks.results import StormtracksResultsManager, StormtracksNumpyResultsManager
    cprint('Using output dir: {}'.format(settings.OUTPUT_DIR), 'green', attrs=['bold'])
    cprint('  Tracking results: {}'.format(settings.TRACKING_RESULTS), 'green', attrs=['bold'])
    results_manager = StormtracksResultsManager(settings.TRACKING_RESULTS)
    results_years = results_manager.list_years()
    for year in results_years:
        if full:
            cprint('    Results year: {}'.format(year), 'blue', attrs=['bold'])
            ensemble_members = results_manager.list_ensemble_members(year)
            for ensemble_member in ensemble_members:
                cprint('      ensemble member: {}'.format(ensemble_member), 'blue')
                for result in results_manager.list_results(year, ensemble_member):
                    if result in ['vort_tracks_by_date', 'good_matches']:
                        cprint('        result: {}'.format(result), 'blue')
        else:
            ensemble_members = results_manager.list_ensemble_members(year)
            vort_tracks_count = 0
            good_matches_count = 0
            for ensemble_member in ensemble_members:
                for result in results_manager.list_results(year, ensemble_member):
                    if result.split('-')[0] == 'vort_tracks_by_date':
                        vort_tracks_count += 1
                    elif result.split('-')[0] == 'good_matches':
                        good_matches_count += 1

            cprint('    Tracking results year: {}; {} ensemble members; {} vort tracks, {} good matches'.
                   format(year, len(ensemble_members), vort_tracks_count, good_matches_count), 'blue', attrs=['bold'])

    cprint('  Field collection results: {}'.format(settings.FIELD_RESULTS), 'green', attrs=['bold'])
    results_manager = StormtracksResultsManager(settings.FIELD_RESULTS)
    results_years = results_manager.list_years()
    for year in results_years:
        if full:
            cprint('    Field coll. esults year: {}'.format(year), 'blue', attrs=['bold'])
            ensemble_members = results_manager.list_ensemble_members(year)
            for ensemble_member in ensemble_members:
                cprint('      ensemble member: {}'.format(ensemble_member), 'blue')
                for result in results_manager.list_results(year, ensemble_member):
                    if result in ['cyclones']:
                        cprint('        result: {}'.format(result), 'blue')
        else:
            ensemble_members = results_manager.list_ensemble_members(year)
            cyclones = 0
            for ensemble_member in ensemble_members:
                for result in results_manager.list_results(year, ensemble_member):
                    if result == 'cyclones':
                        cyclones += 1

            cprint('    Field coll. results year: {}; {} ensemble members; {} cylones'.format(year, len(ensemble_members), cyclones), 'blue', attrs=['bold'])



def main():
    if sys.argv[1] == 'install':
        install()
    elif sys.argv[1] == 'install_full':
        install_aptitude()
        install()
    elif sys.argv[1] == 'clean':
        clean()
    elif sys.argv[1] == 'reinstall':
        clean()
        install()
    elif sys.argv[1] == 'list_data_sources':
        list_data_sources()
    elif sys.argv[1] == 'list_output':
        list_output()


if __name__ == '__main__':
    main()
