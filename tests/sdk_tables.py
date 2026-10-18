"""The DJI Thermal SDK v1.8's own constant tables, read out of its libraries: the reference the
built-in tables (dji_thermal_cli.tsdk.tables) are checked against. Test-only.

Each table is located by a short byte anchor and accepted only if its SHA-256 matches.
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import numpy as np

# name: (library, size in bytes, numpy dtype, 16-byte anchor, sha256 of the table)
TABLES = {
    "target_temp_table": ("iirp", 0x1C0, "<f8", "cdcccccccc046f406666666666127140", "a5a01c0ab5590b05a7735b3c5b62052fccf4639a546a4b60bb6f12ed5ac9f5b7"),
    "dist_table": ("iirp", 0x200, "<f8", "000000000000d03f333333333333d33f", "a2daedd5e0923ed1a5d55a94327d20b5d9369f6b7fcd7e7d798060ede1035b6f"),
    "new_target_temp_table": ("iirp", 0x150, "<f8", "cdcccccccc046f406666666666127140", "81f3188126f009ce5ef3b4510ff5a2580a55cab4bec00dc7db82bc030491d202"),
    "new_dist_table": ("iirp", 0x2C0, "<f8", "000000000000d03f333333333333d33f", "6cb2d598349d32c155825eb46a02b52572f7edc7f2536c259acbbb4530f9b59a"),
    "target_temp_table_v3": ("iirp", 0x168, "<f8", "cdcccccccc046f406666666666127140", "7363c7e0c1bc3412fc9f3a4bedc4830f2291679724c5df3f90c9a4f72fadcfc3"),
    "dist_table_v3": ("iirp", 0x2C0, "<f8", "000000000000d03f333333333333d33f", "6cb2d598349d32c155825eb46a02b52572f7edc7f2536c259acbbb4530f9b59a"),
    "org_ems_list_of_ems_table_v1": ("iirp", 0xA0, "<f8", "9a9999999999e93fec51b81e85ebe93f", "7cc53acdf1a235bfa19a75c2c38daab27a4427091e9ddab38ec159983eedccaa"),
    "target_temp_list_of_ems_table_v1": ("iirp", 0x2B0, "<f8", "cdcccccccce46b40cdcccccccc246d40", "9854cd1e498d6ef25253aa44ff59f5d69f2e02ecf159c6c3693ba03f7dab9210"),
    "ems_correct_table_v1": ("iirp", 0x35C0, "<f8", "000000000000e83fa4703d0ad7a3e83f", "e91791fddfa42eae139469946530104c1a363bda25f67b166bc1ba28d6622a86"),
    "org_ems_list_of_ems_table_v2": ("iirp", 0x320, "<f8", "9a9999999999e93f448b6ce7fba9e93f", "8a3b38dbeaf905ab5c220faa7e069f58e8d85b7749d0c38e96f30f60fc113496"),
    "target_temp_list_of_ems_table_v2": ("iirp", 0x288, "<f8", "cdcccccccce46b40cdcccccccc246d40", "4971aad8072682b01fcc1cfc613839912647a90ce0c3aded24ca5d860d1070e2"),
    "ems_correct_table_v2": ("iirp", 0xFD20, "<f8", "e9263108ac1ce83ff1f44a598638e83f", "6ebe4ae85a7800a238adc48b3efb9b698b335439069490a6bf3a9d2a65920bce"),
    "default_tau_gain0": ("iirp", 0x1C00, "<u2", "00407b3ffa3e803e123eb23d643d213d", "8ac1aa101e64e10b7fdb6564a1b39cd296a8642a68c456749d33a28a1fdf4fcc"),
    "default_tau_gain1": ("iirp", 0x1C00, "<u2", "0040603fc53e333eb03d3f3de63ca03c", "b0a5ce11e3792d0d7f409f42472ff86d9ea5804b0b83ddac173b2ec867b53c47"),
    "girp_luts": ("girp", 0x50000, "<i2", "d4fed4fed4fed4fed4fed4fed4fed4fe", "8a2fb3a0d7df45d438a1a04512ce6ad931d30e52b62169aee1cbbd2d0dc58c62"),
    "girp_palettes": ("girp", 0x2100, "u1", "01010101010101010103030304040405", "72182a72c282634b03bf2dd0f9c5353d65647972e45f277b65025ffe4d93c7fd"),
    "dirp_palettes": ("dirp", 0xF000, "<f8", "00000000000030400000000000c03040", "8ae24eed76a03e5b5d45e0ddea79bad1cd618d38abad8ee36d0b36f36bf0a7f4"),
    "dirp_y2r": ("dirp", 0x60, "<f8", "fe65f7e461a1f23f0000000000000000", "6c19039428f2e12acfb58fa617b3433c19c1a26bc395d14fa74ccf0d889410cc"),
    "isp_yuv_1": ("iirp", 0x300, "u1", "00808001808002808003808004808005", "6d85dc439eb88c4df7a6aaba9597806068be76d20f78313fb524a528ef530c3b"),
    "isp_yuv_3": ("iirp", 0x300, "u1", "04907d04917e05927e05927e06937f06", "1720679824f728b7c7c3d9a1e9afaac1c1ac4b1d8297057e045552303bcfcf2f"),
    "isp_yuv_4": ("iirp", 0x300, "u1", "0fc2750fc47510c67510c87411ca7411", "46013cb4e6d05aecfaa6262e9d0a7ed673bd000a9bc22ca49bd32bb5feb093b0"),
    "isp_yuv_5": ("iirp", 0x300, "u1", "2d8b8d308c8f338d91358f923890953b", "4946cf796b4131b870af34a0f6ac0bdc5d48ef9653e8be3f7d5b6dae8b184614"),
    "isp_yuv_6": ("iirp", 0x300, "u1", "266ac0286dc22a6fc32c72c42e74c630", "1cad602ca20e382bf27f428f05497512807c89897b19d076336bb48a7b2e416d"),
    "isp_yuv_7": ("iirp", 0x300, "u1", "00808001808003808004808006808007", "161d41d8739fce0d4362cb53a96d565638dc4d126f6c780ea3ea7a3755329088"),
    "isp_yuv_8": ("iirp", 0x300, "u1", "00808001808002807f03817f05817e06", "aff108f83501bec7f2e8bc8edc33f5521f126a2e76a4c2788ca47962e95ccf4b"),
    "isp_yuv_9": ("iirp", 0x300, "u1", "1a956d1b986d1d9b6c1e9d6b1fa16a21", "ce99879315d7bd20410aeb7a5311b5e990d0573442e3553744a0c24f533f330b"),
    "isp_yuv_10": ("iirp", 0x300, "u1", "008080017f82017f83027f84037e8604", "deeb1c2273cfd099ae26a9b9a2d80f5005022578e6cbb075408e763dfef60ee5"),
    "isp_yuv_11": ("iirp", 0x300, "u1", "ff8080fe8080fd8080fc8080fb8080fa", "a57b93f8f181e6f471d481d02250840ee2aa7f5dd62654f34e7341c325e27cd4"),
    "isp_yuv_12": ("iirp", 0x300, "u1", "00808001808002808003808004808005", "ec464631d0cefb362ff064708a25a14e017e53e24134a0f88670ad3e3c029cfc"),
    "isp_rgb_1": ("iirp", 0x300, "u1", "00000001010102020203030304040405", "72432263dbfe17abc40ed269f24c7a344e077e3671007dfc8a2f3851f8193dc2"),
    "isp_rgb_3": ("iirp", 0x300, "u1", "00002001002202002403002604002805", "e43edbc491ddd0596e25b67b5e8c60531b75bc16a598db73537654ef373a6f02"),
    "isp_rgb_4": ("iirp", 0x300, "u1", "00008300008700008b00008f00009300", "9f3df695bd7d825058709b1d35177e1f3f5c62c5a6ae8911d2c8e407cec17755"),
    "isp_rgb_5": ("iirp", 0x300, "u1", "4020404521454a224a4f234f5524555a", "d1a40c4b63fb1c8989fa5e8283c6be5c0102bdc94b16e61bb5601b0611229303"),
    "isp_rgb_6": ("iirp", 0x300, "u1", "80000084000688000c8c001390001994", "aa41fdba392af6f4518ecb0672828696b9de2637efd3e1f0448f1d5ae920fd89"),
    "isp_rgb_7": ("iirp", 0x300, "u1", "00000001010103030304040406060607", "a4b5e4684f20fcdd918ab472730cb05565f2e17766fdb8fd8a7f5ee0e396d4ff"),
    "isp_rgb_8": ("iirp", 0x300, "u1", "00000000010101030301040401060602", "cbd8c19f7c659e74bfeff892c3db45ff762ff77c1aa41c7cf4027891a39ceea0"),
    "isp_rgb_9": ("iirp", 0x300, "u1", "00204000214600224c00235200245900", "639463146d4c52e09e63dafdc3e507b0c20f5b29bbe3a2e0aebc453764a1e2d4"),
    "isp_rgb_10": ("iirp", 0x300, "u1", "0000000300000500000800000b00000d", "bba1c721a248c995460e6e019cf11861b8fdf2ff7e60b915eb8034b8b01bf244"),
    "isp_rgb_11": ("iirp", 0x300, "u1", "fffffffefefefdfdfdfcfcfcfbfbfbfa", "346f6a25ad11ec5b45a83392366f269058ba209d2877491634a2405f86beb3db"),
    "isp_snr_weights": ("iirp", 0x300, "<i4", "0c000000000000000000000000000000", "740eab51ecf2db2d460d8149becfa587038045d33967a6f9226111c10a79b7bb"),
    "isp_gauss3": ("iirp", 0x24, "<i4", "0d000000240000000d00000024000000", "6d7efbdf27678d622b13d91012398cf3b057abb372f1fede113a9206a1412f74"),
    "mt_exp_frac": ("microta", 0x8000, "<u2", "00400140024003400440054006400740", "18d11a78bf085cb7353b056c17e2bdbba6bb0f86a364729e5eff02ba01f4d438"),
    "mt_sqrt": ("microta", 0x800, "<u2", "a116a316a616a916ac16af16b216b416", "b86844044b56d53bf35176d421699a6d477077fbf2a5f552dd1dcf72bc68e4b0"),
    "mt_sqrt64_a": ("microta", 0x400, "<i8", "7dd50200000000004fd8020000000000", "6b9d28d5197c4e03b84a02eb6c0bba200e2e786a9be45b6bdf0b768250c0d6ae"),
    "mt_sqrt64_b": ("microta", 0x400, "<i8", "abd2020000000000dfcf020000000000", "72512a5e1d764cd322a3ee3404e5b90eac2e1070fd4aa0edfc5aa08da143dc29"),
    "mt_inv_a": ("microta", 0x800, "<i8", "c07f000000000000417f000000000000", "babcc9cc864a6047b855b4da29d659013ad4f43d1f9996baa43863d269b7a791"),
    "mt_inv_b": ("microta", 0x800, "<i8", "807f000000000000837e000000000000", "e1eb23b251d49c365baea260ae3e85efc6a36262d5db512b136b2e5d0287efa8"),
    "mt_log": ("microta", 0x800, "<u2", "00001000200030004000500060007000", "5870706812e0642d5674a6d0cdf3e98724461039181b572eb5a57cc97fdcbc4b"),
    "mt_atm_k": ("microta", 0x28, "<i4", "00400000e53f00004c3f0000c93e0000", "ddf6b6b8cf459765b2d2053d4ddcf32b898d6465eee1a94dc69f519253818c23"),
    "mt_atm_b": ("microta", 0x28, "<i4", "cdccfeffefeefeff643bffff295cffff", "6625187c04e9fa28be0ac250d8883e861ea6394d00b019c7f034b3550efe2abb"),
    "mt_atm_lim": ("microta", 0x28, "<i4", "03000000080000001000000020000000", "7ade0df1fe41bf17c37070c29efad4278afe360f7049d26b96f82de0ead7dcb9"),
}
_LIB_NAMES = {
    "iirp": ("libv_iirp.so", "libv_iirp.dll"),
    "girp": ("libv_girp.so", "libv_girp.dll"),
    "microta": ("libMicroTA_Release_x64.so", "libMicroTA_Release_x86.so"),
    "dirp": ("libv_dirp.so", "libv_dirp.dll"),
}


def _library_blobs(source: Path):
    """Yield (lib kind, bytes) for every SDK library holding tables (libv_iirp, libv_girp, libv_dirp, libMicroTA) in a zip, dir or file."""
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            for name in z.namelist():
                base = name.rsplit("/", 1)[-1]
                for kind, names in _LIB_NAMES.items():
                    if base in names:
                        yield kind, z.read(name)
        return
    if source.is_file():
        for kind, names in _LIB_NAMES.items():
            if source.name in names:
                yield kind, source.read_bytes()
        return
    if source.is_dir():
        for kind, names in _LIB_NAMES.items():
            for name in names:
                for path in sorted(source.rglob(name)):
                    yield kind, path.read_bytes()


def _find(blob: bytes, size: int, anchor: bytes, digest: str) -> bytes | None:
    i = blob.find(anchor)
    while i != -1:
        candidate = blob[i:i + size]
        if len(candidate) == size and hashlib.sha256(candidate).hexdigest() == digest:
            return candidate
        i = blob.find(anchor, i + 1)
    return None


def extract(source: str | Path) -> dict[str, np.ndarray]:
    """Every table found in `source`, by name."""
    found: dict[str, np.ndarray] = {}
    for kind, blob in _library_blobs(Path(source)):
        for name, (lib, size, dtype, anchor, digest) in TABLES.items():
            if lib == kind and name not in found:
                raw = _find(blob, size, bytes.fromhex(anchor), digest)
                if raw is not None:
                    found[name] = np.frombuffer(raw, dtype=dtype).copy()
    return found
