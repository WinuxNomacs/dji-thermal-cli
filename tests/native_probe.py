"""Subprocess helper for the tests: the API revision the native DJI Thermal SDK reports.

    native_probe.py <file> [<file> ...]

Prints JSON {"revisions": [...], "magic": "..."}: the distinct revisions reported for the
files (the revision is reported per camera plugin), or an "error" when the SDK can't open them.
"""

import json
import os
import sys

os.environ["DJI_THERMAL_SDK_ALLOW_UNKNOWN_VERSION"] = "1"     # report, don't refuse
import native_sdk as sdk  # noqa: E402

revisions, magic, errors = set(), "", []
lib = sdk.DirpSDK()
for name in sys.argv[1:]:
    try:
        with open(name, "rb") as f, lib.open(f.read()) as r:
            found = lib.api_version(r)
    except sdk.DirpError as e:
        errors.append(f"{name}: {e}")
        continue
    if found is not None:
        revisions.add(found[0])
        magic = found[1]
print(json.dumps({"revisions": sorted(revisions), "magic": magic, "errors": errors}))
