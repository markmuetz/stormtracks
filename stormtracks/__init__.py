from .version import __version__
try:
    from .c20data import C20Data
    from .ibtracsdata import IbtracsData
    __all__ = [
	'__version__',
	'C20Data',
	'IbtracsData',
    ]
except (ImportError, OSError):
    # This can be called on installation, when numpy et al. won't be installed.
    # Handle import errors and just expose __version__.
    __all__ = [
	'__version__',
    ]
