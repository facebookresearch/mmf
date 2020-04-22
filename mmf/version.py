import sys

__version__ = "0.9.alpha2"

msg = "MMF is only compatible with Python 3.6 and newer."


if sys.version_info < (3, 6):
    raise ImportError(msg)
