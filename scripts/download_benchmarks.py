"""Download the raw files of the external benchmarks from their original hosts.

- CASMI 2016 and EMBL-MCF 2.0: the SDF library exports of MassBank of North
  America (MoNA). Their export ids are looked up in MoNA's catalog of
  predefined downloads, so a re-export does not break the script.
- CASMI 2017: challenges 1-45 and the solutions from casmi-contest.org, which
  serves them over http only.

Files are written in the layout scripts/casmi2mgf.py and scripts/embl2mgf.py
read. Files already present are kept unless --force is given. manifest.json
records the URL, size, SHA-256 and date of every file downloaded, since the
hosts can re-export a library with other spectra, and warns when a file
differs from the one behind the documentation's figures (EXPECTED).

Usage:
    python scripts/download_benchmarks.py --out_dir ./data/benchmark
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import urllib.request
import zipfile

MONA = "https://mona.fiehnlab.ucdavis.edu/rest/downloads"
CASMI2017 = "http://www.casmi-contest.org/2017"
USER_AGENT = "msfiddle-benchmark-downloader"

# MoNA catalog label -> output path
MONA_LIBRARIES = {
    "Libraries - MassBank - CASMI 2016": "casmi2016/MoNA-export-CASMI_2016.sdf",
    "Libraries - EMBL-MCF_2.0_HRMS_Library": (
        "embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf"
    ),
}
CASMI2017_FILES = {
    f"{CASMI2017}/challenges/Chal1to45Summary.csv": "Chal1to45Summary.csv",
    f"{CASMI2017}/CASMI-solutions.csv": "CASMI-solutions.csv",
}
# Fingerprints of the files the documentation's figures were computed from
# (downloaded 2026-11-18). SDF fingerprints skip the per-record header line
# with the export timestamp, so a re-export of the same spectra still matches.
EXPECTED = {
    "casmi2016/MoNA-export-CASMI_2016.sdf": "0110c19add0ae85703fd5ab813d1632f1a01b7bbc7040bd6dcc94ae7e5c91923",
    "embl/MoNA-export-EMBL-MCF_2.0_HRMS_Library.sdf": "62382b92f444dd0befdd38c907092af2cb16a722a8993878427a25f64bbd8c1e",
    "casmi2017/Chal1to45Summary.csv": "0d39cb0e191079f6e87af99d4aa67cb4b5faf7aff2a2607356ce26dfcf1e935d",
    "casmi2017/CASMI-solutions.csv": "d1aed33a27746999fcf60ad0ff93389a515a8e012c7f66b028a34aa5528542a6",
    "casmi2017/challenges-001-045-msms-mgf-20170908": "e2a8e09fa9527a62902bd5020301bed9062e5ec26d1538d5e92e89ab81d7cad1",
    "casmi2017/challenges-001-045-ms-mgf-20170908": "bd4cd34560ef2b82f5a0df828fa40a5a6d39403d258b0b5c32954d37e1e65179",
}
CASMI2017_ZIPS = [
    "challenges-001-045-msms-mgf-20170908",
    "challenges-001-045-ms-mgf-20170908",
]


MANIFEST = {}


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    MANIFEST[url] = {
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "downloaded": datetime.date.today().isoformat(),
    }
    return data


def fingerprint(path):
    """SHA-256 of a file, or of a folder's files by name; for an SDF, without
    the second line of each record (program and export timestamp)."""
    digest = hashlib.sha256()
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            digest.update(name.encode())
            digest.update(bytes.fromhex(fingerprint(os.path.join(path, name))))
        return digest.hexdigest()
    with open(path, "rb") as f:
        data = f.read()
    if path.endswith(".sdf"):
        records = data.split(b"$$$$")
        data = b"$$$$".join(
            b"\n".join(
                line for i, line in enumerate(r.lstrip(b"\r\n").split(b"\n")) if i != 1
            )
            for r in records
        )
    digest.update(data)
    return digest.hexdigest()


def check_expected(out_dir):
    """Warn about files that differ from those behind the documentation."""
    changed = [
        rel
        for rel, expected in EXPECTED.items()
        if os.path.exists(os.path.join(out_dir, rel))
        and fingerprint(os.path.join(out_dir, rel)) != expected
    ]
    for rel in changed:
        print(
            f"Warning: {rel} differs from the file the documentation's results "
            "were computed from; the host may have re-exported it, so results "
            "can differ from the documentation."
        )
    if not changed:
        print("All files match those behind the documentation's results.")
    return changed


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)


def extract_flat(data, out_dir, suffix=None):
    """Extract a zip's files into out_dir by base name (archives keep them in a
    subfolder); returns the names written."""
    names = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            name = os.path.basename(info.filename)
            if info.is_dir() or not name or (suffix and not name.endswith(suffix)):
                continue
            write(os.path.join(out_dir, name), archive.read(info))
            names.append(name)
    return names


def download_mona(out_dir, force):
    catalog = {
        entry["label"]: entry for entry in json.loads(fetch(f"{MONA}/predefined"))
    }
    for label, rel_path in MONA_LIBRARIES.items():
        path = os.path.join(out_dir, rel_path)
        if os.path.exists(path) and not force:
            print(f"Keep {path}")
            continue
        if label not in catalog:
            raise SystemExit(f"MoNA no longer lists {label!r}; check {MONA}/predefined")
        export = catalog[label]["sdfExport"]
        data = fetch(f"{MONA}/retrieve/{export['id']}")
        if export.get("size") and len(data) != int(export["size"]):
            raise SystemExit(
                f"{label}: got {len(data)} bytes, MoNA lists {export['size']}"
            )
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            sdf = [n for n in archive.namelist() if n.endswith(".sdf")]
            if len(sdf) != 1:
                raise SystemExit(f"{label}: expected one SDF in the export, got {sdf}")
            write(path, archive.read(sdf[0]))
        print(f"Saved {path} ({export.get('count', '?')} spectra)")


def download_casmi2017(out_dir, force):
    base = os.path.join(out_dir, "casmi2017")
    for url, name in CASMI2017_FILES.items():
        path = os.path.join(base, name)
        if os.path.exists(path) and not force:
            print(f"Keep {path}")
            continue
        write(path, fetch(url))
        print(f"Saved {path}")
    for stem in CASMI2017_ZIPS:
        folder = os.path.join(base, stem)
        if os.path.isdir(folder) and os.listdir(folder) and not force:
            print(f"Keep {folder}/")
            continue
        names = extract_flat(fetch(f"{CASMI2017}/challenges/{stem}.zip"), folder)
        print(f"Saved {len(names)} files to {folder}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out_dir", default="./data/benchmark")
    parser.add_argument(
        "--force", action="store_true", help="Download files already present again"
    )
    args = parser.parse_args()
    download_mona(args.out_dir, args.force)
    download_casmi2017(args.out_dir, args.force)
    check_expected(args.out_dir)
    if MANIFEST:
        path = os.path.join(args.out_dir, "manifest.json")
        old = json.load(open(path)) if os.path.exists(path) else {}
        old.update(MANIFEST)
        with open(path, "w") as f:
            json.dump(old, f, indent=2)
        print(f"Recorded the downloads in {path}")
