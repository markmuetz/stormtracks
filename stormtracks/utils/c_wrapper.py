import os
import ctypes
from numpy.ctypeslib import ndpointer

local_dir = os.path.dirname(os.path.abspath(__file__))

# Find the library.
stormtracks_lib = ctypes.cdll.LoadLibrary(os.path.join(local_dir, "../../stormtracks.so"))

cvort = stormtracks_lib.cvort
cvort.restype = None
cvort.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ctypes.c_size_t,
                  ctypes.c_size_t,
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ctypes.c_float,
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

cvort4 = stormtracks_lib.cvort4
cvort4.restype = None
cvort4.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ctypes.c_size_t,
                   ctypes.c_size_t,
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ctypes.c_float,
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]

cextrema = stormtracks_lib.cextrema
cextrema.restype = None
cextrema.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                     ctypes.c_size_t,
                     ctypes.c_size_t,
                     ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
