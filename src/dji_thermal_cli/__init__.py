"""dji-thermal-cli: DJI R-JPEG thermal images to temperatures, in pure Python.

A command line (`dji-thermal-cli`) and the same functionality as a Python API:

    import dji_thermal_cli as dtc

    dtc.check("flight/").findings
    dtc.measure("flight/DJI_0001_T.JPG", unit="F").temperature
    dtc.process("flight/", "out/", dtc.Settings(emissivity=0.95), outputs=["temp_c"])

See dji_thermal_cli.api for everything available; dji_thermal_cli.tsdk is the port of DJI's
Thermal SDK v1.8 underneath.
"""

from ._version import __version__
from .api import *  # noqa: F403 - the public API, listed in api.__all__
from .api import __all__ as _api_all
from .cli import main

__all__ = [*_api_all, "main", "__version__"]
