"""The package version, read from the installed distribution (pyproject.toml is its one source)."""

from importlib import metadata

try:
    __version__ = metadata.version("dji-thermal-cli")
except metadata.PackageNotFoundError:          # a source tree that isn't installed
    __version__ = "0+unknown"
