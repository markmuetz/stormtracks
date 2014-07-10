import os, sys

def check_dependencies():
    is_missing_module = False
    for module_name in ['numpy', 'scipy', 'pylab', 'netCDF4', 'mpl_toolkits.basemap', 'stormtracks']:
        try:
            mymodule = __import__(module_name)
        except:
            is_missing_module = True
            print('Module {0} not found\n'.format(module_name))
            if module_name == 'stormtracks':
                stormtracks_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                print('''You can add stormtracks to your local path (temporarily):

export PYTHONPATH=$PYTHONPATH:{0}
'''.format(stormtracks_dir))
                print('''Alternatively you can install stormtracks by going up a directory and running

python setup.py install
''')
            else:
                if module_name == 'pylab':
                    pip_name = 'matplotlib'
                elif module_name == 'mpl_toolkits.basemap':
                    pip_name = 'basemap'
                else:
                    pip_name = module_name

                print('Please install the module through pip or your local package manager, e.g.\n')
                print('pip install {0}\n'.format(pip_name))

    if not is_missing_module:
        print('All dependencies present')
    return not is_missing_module

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit('Not all dependencies installed')

