"""The derivative products `process` can write, their pixel encodings, and the file writers.

A product is what the pixels mean (temperature in some encoding, raw counts, a view); a
format is the file type it is written as. An output is a (product, format) pair, given on
the command line as PRODUCT[:FORMAT] and written as <stem>_<product>.<format>.

TIFF and EXR are compressed losslessly by default (Deflate / ZIP) or not at all
(compression "none"); PNG is always zlib-compressed.

Every TIFF and PNG carries the source EXIF (GPS included) and XMP; single-band thermal
products also get Camera:BandName="LWIR" for WebODM/ODX.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from . import exrout, files, tiffout, units

KELVIN = units.KELVIN


@dataclasses.dataclass(frozen=True)
class Product:
    description: str
    formats: tuple[str, ...]        # first: the default format


PRODUCTS = {
    "temp_c": Product("float32 temperature in degrees C (no data: -273.15)", ("tiff", "exr")),
    "temp_f": Product("float32 temperature in degrees F (no data: -459.67)", ("tiff", "exr")),
    "temp_k": Product("float32 temperature in kelvin (no data: 0)", ("tiff", "exr")),
    "temp_ck": Product("uint16 temperature in centi-kelvin, K x 100 (0.01 K steps up to 382 C; no data: 0)", ("tiff", "png")),
    "temp_dc": Product("int16 temperature in tenths of a degree C, the DJI SDK's dirp_measure output", ("tiff",)),
    "raw": Product("uint16 raw sensor counts, uncalibrated (dirp_get_original_raw)", ("tiff", "png")),
    "gray16": Product("uint16 grayscale view: --range-min..--range-max (or each image's own range) "
                      "stretched to 0..65535; for viewing, not measurement", ("png", "tiff")),
    "pseudo": Product("8-bit RGB pseudo-colour image (--palette, --range-min/--range-max)", ("png", "tiff")),
}
FORMAT_ALIASES = {"tif": "tiff"}
DEFAULT_OUTPUTS = (("temp_c", "tiff"), ("pseudo", "png"))
COMPRESSIONS = ("deflate", "none")


def parse_output(spec: str) -> tuple[str, str]:
    """PRODUCT[:FORMAT] -> (product, file extension). Raises ValueError with the choices."""
    product, _, ext = spec.strip().partition(":")
    if product not in PRODUCTS:
        raise ValueError(f"unknown product {product!r}; choose from {', '.join(PRODUCTS)}")
    formats = PRODUCTS[product].formats
    ext = ext.lower() or formats[0]
    if FORMAT_ALIASES.get(ext, ext) not in formats:
        allowed = ", ".join(f + ("/tif" if f == "tiff" else "") for f in formats)
        raise ValueError(f"{product} can be written as {allowed}, not {ext!r}")
    return product, ext


def file_name(stem: str, product: str, ext: str) -> str:
    return f"{stem}_{product}.{ext}"


# --- encodings ------------------------------------------------------------------------------

def centikelvin(temp_c: np.ndarray) -> tuple[np.ndarray, int]:
    """uint16 K x 100 from Celsius, and the count of pixels clipped to the uint16 range.
    No-data pixels (-273.15 C) become 0."""
    v = np.rint((temp_c.astype(np.float64) + KELVIN) * 100.0)
    clipped = int(np.count_nonzero(v > 65535))
    return np.clip(v, 0, 65535).astype(np.uint16), clipped


def gray16(temp_c: np.ndarray, low: float | None, high: float | None) -> tuple[np.ndarray, float, float]:
    """uint16 view of low..high (default: the image's valid range) stretched to 0..65535,
    and the range used."""
    valid = temp_c[temp_c > tiffout.NO_DATA_C]
    if low is None or high is None:
        low, high = (float(valid.min()), float(valid.max())) if valid.size else (0.0, 0.0)
    if high <= low:
        return np.zeros(temp_c.shape, np.uint16), low, high
    v = np.rint((temp_c.astype(np.float64) - low) / (high - low) * 65535.0)
    return np.clip(v, 0, 65535).astype(np.uint16), low, high


# --- writers --------------------------------------------------------------------------------

def write(path: Path, product: str, ext: str, pixels: np.ndarray, metadata: tiffout.SourceMetadata,
          text: dict[str, str] | None = None, compression: str = "deflate") -> None:
    """Write one output. text: extra key/value metadata (PNG text chunks, EXR attributes).
    compression: "deflate" (TIFF Deflate, EXR ZIP) or "none"; PNG is always compressed."""
    fmt = FORMAT_ALIASES.get(ext, ext)
    if fmt not in ("tiff", "png", "exr"):
        raise ValueError(fmt)
    band_name = product != "pseudo"
    with files.replaced_when_done(path) as tmp:
        if fmt == "tiff":
            tiffout.write_tiff(tmp, pixels, metadata, band_name=band_name, compression=compression)
        elif fmt == "png":
            _write_png(tmp, pixels, metadata, band_name, text or {})
        else:
            exrout.write_exr(tmp, pixels, text, compression="zip" if compression == "deflate" else "none")


def _write_png(path: Path, pixels: np.ndarray, metadata: tiffout.SourceMetadata, band_name: bool,
               text: dict[str, str]) -> None:
    """8-bit RGB or 16-bit grayscale PNG with the source EXIF (eXIf) and XMP (iTXt)."""
    info = PngImagePlugin.PngInfo()
    xmp = tiffout.with_band_name(metadata.xmp) if band_name else metadata.xmp
    if xmp:
        info.add_itxt("XML:com.adobe.xmp", xmp.decode("utf-8", "replace"))
    for key, value in text.items():
        info.add_text(key, value)
    kwargs = {"pnginfo": info}
    exif = tiffout.exif_for(metadata, pixels.shape[1], pixels.shape[0])
    if exif:
        kwargs["exif"] = exif
    image = Image.fromarray(pixels)                 # mode from the array: RGB, L or I;16
    image.save(path, **kwargs)
