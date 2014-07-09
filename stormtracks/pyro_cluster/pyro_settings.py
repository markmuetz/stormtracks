import socket

hostname = socket.gethostname()

if len(hostname.split('.')) >= 2:
    is_ucl = True
    manager = 'madrid'
    worker_servers = [
    'berlin', 
    'warsaw', 
    'naples', 
    'vienna', 
    'venice', 
    'prague', 
    #'lisbon', 
    'dublin', 
    'rome', 
    'oslo', 
    'athens', 
    'cologne', 
    'mainz', 
    'linz', 
    'seville', 
    'bergen', 
    'granada', 
    'cordoba', 
    'zurich', 
    'leipzig', 
    'riga', 
    'vilnius', 
    'tallinn', 
    'salzburg', 
    'antwerp', 
    'budapest', 
    'helsinki', 
    'munich', 
    'nice', 
    'turin', 
    'marseille'] 
else:
    is_ucl = False
    manager = '192.168.0.15'
    worker_servers = ['192.168.0.2']
