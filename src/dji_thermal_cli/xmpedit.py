"""Small, targeted edits of an XMP packet: set simple properties of one namespace, or remove
properties by name.

Properties are written as attributes of the first rdf:Description (declaring the namespace
there if the packet doesn't), or updated in place where the packet already has them, as an
attribute (prefix:Name="...") or as an element (<prefix:Name>...</prefix:Name>). Everything
else in the packet is kept byte for byte.
"""

from __future__ import annotations

import re
from xml.sax.saxutils import escape, quoteattr

DRONE_DJI = ("drone-dji", "http://www.dji.com/drone-dji/1.0/")
DTC = ("dtc", "urn:dji-thermal-cli:xmp:1.0")                  # what this tool did to an output

_EMPTY = ('<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
          '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
          '<rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


def set_properties(xmp: bytes | None, namespace: tuple[str, str], props: dict[str, str]) -> bytes:
    """The packet with each prop set to its value (str), in namespace (prefix, uri)."""
    prefix, uri = namespace
    if not xmp:
        text, encoding = _EMPTY, "utf-8"
    else:
        try:
            text, encoding = xmp.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            text, encoding = xmp.decode("latin-1"), "latin-1"
        if "<rdf:Description" not in text:
            text, encoding = _EMPTY, "utf-8"
    added = []
    for name, value in props.items():
        attr = re.compile(rf'(\b{re.escape(prefix)}:{re.escape(name)}\s*=\s*)("[^"]*"|\'[^\']*\')')
        elem = re.compile(rf'(<{re.escape(prefix)}:{re.escape(name)}>)([^<]*)(</{re.escape(prefix)}:{re.escape(name)}>)')
        if attr.search(text):
            text = attr.sub(lambda m: m.group(1) + quoteattr(value), text, count=1)
        elif elem.search(text):
            text = elem.sub(lambda m: m.group(1) + escape(value) + m.group(3), text, count=1)
        else:
            added.append(f" {prefix}:{name}={quoteattr(value)}")
    if added:
        declare = "" if f"xmlns:{prefix}=" in text else f' xmlns:{prefix}="{uri}"'
        text = text.replace("<rdf:Description", "<rdf:Description" + declare + "".join(added), 1)
    return text.encode(encoding)


_ATTR = re.compile(r'\s+([A-Za-z_][\w.-]*):([A-Za-z_][\w.-]*)\s*=\s*("[^"]*"|\'[^\']*\')')
_START = re.compile(r'<([A-Za-z_][\w.-]*):([A-Za-z_][\w.-]*)[\s/>]')


def remove_properties(xmp: bytes, remove) -> bytes:
    """The packet without every property whose local name satisfies remove(name), in any
    namespace: attributes (prefix:Name="..."), and elements with their content
    (<prefix:Name>...</prefix:Name>, including structured values). Namespace declarations and
    everything else are kept."""
    try:
        text, encoding = xmp.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, encoding = xmp.decode("latin-1"), "latin-1"
    text = _ATTR.sub(lambda m: "" if m.group(1) != "xmlns" and remove(m.group(2)) else m.group(0), text)
    for prefix, name in {m.groups() for m in _START.finditer(text)}:
        if prefix not in ("rdf", "x") and remove(name):
            q = f"{re.escape(prefix)}:{re.escape(name)}"
            text = re.sub(rf"<{q}(?:\s[^>]*?)?(?:/>|>.*?</{q}\s*>)", "", text, flags=re.S)
    return text.encode(encoding)
