"""Exceptions raised by the pure-Python DJI Thermal SDK port."""


class TsdkError(Exception):
    """Base class. `code` mirrors the native SDK's dirp_ret_code_e where one applies."""

    code = -1


class RJpegError(TsdkError, ValueError):
    """The data is not an R-JPEG this port can parse."""

    code = -7


class UnsupportedRJpeg(RJpegError):
    """A thermal R-JPEG whose format the pure-Python port does not (yet) implement: a DJI format,
    or another maker's (FLIR's radiometric JPEG, as DJI's Zenmuse XT2 writes)."""


class ParamError(TsdkError, ValueError):
    """Measurement parameters the camera's backend rejects (outside its range, or unusable)."""

    def __init__(self, message: str, code: int = -3):
        super().__init__(message)
        self.code = code


class MeasureError(TsdkError):
    """The backend refused to measure (the native SDK fails the same way)."""

    def __init__(self, message: str, code: int = -32):
        super().__init__(message)
        self.code = code


class IspError(TsdkError):
    """The image pipeline (pseudo-colour rendering) failed the way the native SDK does."""

    def __init__(self, message: str, code: int = -32):
        super().__init__(message)
        self.code = code


class UndefinedBehaviour(IspError):
    """The native code would read outside a table or divide by zero here (undefined result)."""

    def __init__(self, message: str):
        super().__init__(message, -32)


class UnsupportedFeature(TsdkError):
    """The pure-Python port does not implement this for the camera's plugin (yet)."""

    code = -9
