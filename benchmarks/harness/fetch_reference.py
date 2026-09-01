#!/usr/bin/env python3
"""Download the expert reference GeoJSON layers from the Yandex.Disk links in
exp_data.csv into benchmarks/data/gold/reference/, and write a manifest.

Two kinds of link in the "Ссылка на облако" column:
  * public  ``https://disk.yandex.ru/d/<key>``          — fetched with NO auth via
    the public API (folder is downloaded as a zip and extracted);
  * private ``https://disk.yandex.ru/client/disk/<path>`` — needs the owner's
    Yandex OAuth token (``YANDEX_OAUTH_TOKEN`` env). Without a token these are
    listed in the manifest as ``needs_token`` and skipped; the simplest fix is to
    re-share those folders as public ``/d/`` links.

Each unique link is a *project folder* containing the layers for several
scenario_ids; the manifest maps every gold row (scenario_id + base_index) to the
local folder. geometry_eval.py then matches individual files to records by name.

Resumable: a link whose local folder already has files is skipped.

Usage:
    python benchmarks/harness/fetch_reference.py            # public links only
    YANDEX_OAUTH_TOKEN=... python benchmarks/harness/fetch_reference.py   # + private
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

_URL_RE = re.compile(r"https?://[^\s,;]+")
_WINDOWS_INVALID = re.compile(r'[<>:"/\\|?*]')

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def extract_urls(cell: str) -> list[str]:
    """A cloud cell may hold several links (one per line). Return them all,
    stripped of trailing punctuation."""
    return [u.rstrip(".,;") for u in _URL_RE.findall(str(cell))]


import pandas as pd
import requests

PUBLIC_META = "https://cloud-api.yandex.net/v1/disk/public/resources"
PUBLIC_API = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
PRIVATE_META = "https://cloud-api.yandex.net/v1/disk/resources"
PRIVATE_API = "https://cloud-api.yandex.net/v1/disk/resources/download"
COL_LINK = "Ссылка на облако"
COL_SID = "scenario_id"


def link_kind(link: str) -> str:
    if "/d/" in link:
        return "public"
    if "/client/disk" in link:
        return "private"
    return "other"


def link_id(link: str) -> str:
    if "/d/" in link:
        key = link.split("/d/", 1)[1].strip().strip("/")
    else:
        key = unquote(link).split("/client/disk/", 1)[-1]
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", key).strip("_")[:80] or "link"


def _save_resource(href: str, dest: Path, is_file: bool, name: str) -> int:
    """A Yandex resource download is a zip for a folder, or the raw bytes for a
    single file. Handle both."""
    dest.mkdir(parents=True, exist_ok=True)
    r = requests.get(href, timeout=180)
    r.raise_for_status()
    if is_file:
        fname = name or "layer.geojson"
        if not fname.lower().endswith((".geojson", ".json")):
            fname += ".geojson"
        (dest / fname).write_bytes(r.content)
        return 1
    n = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            # Yandex folders may contain ':' and backslashes, which are legal in
            # the archive but not in Windows path components.  Extract manually,
            # normalise every component, and reject traversal instead of relying
            # on ZipFile.extract's platform-dependent path handling.
            archive_path = PurePosixPath(info.filename.replace("\\", "/"))
            parts = []
            for part in archive_path.parts:
                if part in {"", "."}:
                    continue
                if part == "..":
                    raise ValueError(f"unsafe archive member: {info.filename}")
                cleaned = _WINDOWS_INVALID.sub("_", part).rstrip(". ") or "_"
                parts.append(cleaned)
            target = dest.joinpath(*parts)
            if not target.resolve().is_relative_to(dest.resolve()):
                raise ValueError(f"unsafe archive member: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            n += 1
    return n


def fetch_public(link: str, dest: Path) -> int:
    meta = requests.get(
        PUBLIC_META, params={"public_key": link, "limit": 10000}, timeout=60
    )
    meta.raise_for_status()
    m = meta.json()
    href = requests.get(PUBLIC_API, params={"public_key": link}, timeout=60).json()[
        "href"
    ]
    return _save_resource(href, dest, m.get("type") == "file", m.get("name", ""))


def fetch_private(link: str, dest: Path, token: str) -> int:
    # the /client/disk/<path> URL segment (url-decoded) is the Disk path
    path = "/" + unquote(link).split("/client/disk/", 1)[-1].split("?")[0]
    hdr = {"Authorization": f"OAuth {token}"}
    m = requests.get(
        PRIVATE_META, params={"path": path}, headers=hdr, timeout=60
    ).json()
    href = requests.get(
        PRIVATE_API, params={"path": path}, headers=hdr, timeout=60
    ).json()["href"]
    return _save_resource(href, dest, m.get("type") == "file", m.get("name", ""))


def has_files(d: Path) -> bool:
    return d.exists() and any(d.rglob("*.geojson"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default="benchmarks/data/gold/exp_data.csv")
    ap.add_argument("--out", default="benchmarks/data/gold/reference")
    ap.add_argument("--token", default=os.getenv("YANDEX_OAUTH_TOKEN", ""))
    ap.add_argument("--limit", type=int, default=0, help="first N unique links (debug)")
    args = ap.parse_args()

    df = pd.read_csv(args.gold, sep=";", engine="python").reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, r in df.iterrows():
        cell = r.get(COL_LINK)
        if not isinstance(cell, str) or not cell.strip():
            continue
        for link in extract_urls(cell):  # a cell may carry several links
            rows.append(
                {
                    "base_index": i,
                    "scenario_id": r.get(COL_SID),
                    "project": r.get("Наименование проекта"),
                    "link": link,
                    "kind": link_kind(link),
                    "local_dir": f"{link_kind(link)}_{link_id(link)}",
                }
            )
    man = pd.DataFrame(rows)
    uniq = man.drop_duplicates("link")
    if args.limit:
        uniq = uniq.head(args.limit)
    print(
        f"{len(man)} gold rows, {len(uniq)} unique links "
        f"({(uniq.kind=='public').sum()} public, "
        f"{(uniq.kind=='private').sum()} private)"
    )

    status = {}
    for _, u in uniq.iterrows():
        dest = out / u["local_dir"]
        if has_files(dest):
            status[u["link"]] = "cached"
            print(f"  cached  {u['local_dir']}")
            continue
        time.sleep(0.3)  # be polite to the public API
        try:
            if u["kind"] == "public":
                n = fetch_public(u["link"], dest)
                status[u["link"]] = f"ok:{n}"
                print(f"  ok {n:3d}f  {u['local_dir']}")
            elif u["kind"] == "private" and args.token:
                n = fetch_private(u["link"], dest, args.token)
                status[u["link"]] = f"ok:{n}"
                print(f"  ok {n:3d}f  {u['local_dir']} (private)")
            else:
                status[u["link"]] = "needs_token"
                print(f"  SKIP     {u['local_dir']} (private, no token)")
        except Exception as e:  # noqa: BLE001
            status[u["link"]] = f"error:{type(e).__name__}"
            print(f"  ERROR    {u['local_dir']}: {e}")

    man["status"] = man["link"].map(status).fillna("skipped")
    man_path = out / "manifest.csv"
    man.to_csv(man_path, index=False)
    ok = sum(1 for s in status.values() if s.startswith(("ok", "cached")))
    need = sum(1 for s in status.values() if s == "needs_token")
    print(f"\nmanifest -> {man_path}")
    print(
        f"links: {ok} downloaded/cached, {need} need a Yandex token, "
        f"{len(status)-ok-need} errored"
    )
    if need:
        print(
            "  → provide YANDEX_OAUTH_TOKEN, or re-share those folders as "
            "public /d/ links, then re-run."
        )


if __name__ == "__main__":
    main()
