import sys
sys.path.insert(0, '/home/ucfamue/DATA/.local/lib/python2.7/site-packages/')

import Pyro4

#Pyro4.naming.main(['-n', 'madrid'])
Pyro4.naming.main(['-n', '192.168.0.15'])
