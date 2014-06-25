import ctypes
from numpy.ctypeslib import ndpointer

libcvort = ctypes.cdll.LoadLibrary("./lib/libcvort.so")
cvort = libcvort.cvort
cvort.restype = None
cvort.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                  ctypes.c_size_t,
                  ctypes.c_size_t,
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
		  ctypes.c_float,
                  ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
cvort4 = libcvort.cvort4
cvort4.restype = None
cvort4.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                   ctypes.c_size_t,
                   ctypes.c_size_t,
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
		   ctypes.c_float,
                   ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
