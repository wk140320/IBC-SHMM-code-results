#!/usr/bin/env python3
"""Download the two cited UCI archives, verify exact hashes, and safely extract them."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_Official_Real_Datasets"
SOURCES = [
    ("HAPT", "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip", "uci_hapt_id341.zip", "4ac4ae064227c07045a99551876b54d204837e995e298b6488559e209b3deb09"),
    ("HARTH", "https://archive.ics.uci.edu/static/public/779/harth.zip", "uci_harth_id779.zip", "2ab54d6b11467ddbe998dfb60355f9f1ae181b0af7a89e7c78bb7af2cf16b64a"),
]

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def safe_extract(archive: Path, target: Path) -> None:
    target_resolved=target.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            dest=(target/info.filename).resolve()
            if dest != target_resolved and target_resolved not in dest.parents:
                raise ValueError(f"Unsafe archive member path: {info.filename}")
        zf.extractall(target)

def main() -> None:
    for dataset,url,filename,expected in SOURCES:
        ds_dir=DATA/dataset
        ds_dir.mkdir(parents=True,exist_ok=True)
        archive=ds_dir/filename
        if not archive.exists() or digest(archive)!=expected:
            fd,tmp=tempfile.mkstemp(prefix=filename+".",dir=ds_dir)
            os.close(fd)
            try:
                print(f"Downloading {dataset} from the official UCI repository...")
                urllib.request.urlretrieve(url,tmp)
                got=digest(Path(tmp))
                if got!=expected:
                    raise ValueError(f"{dataset} archive SHA-256 mismatch: expected {expected}, got {got}")
                os.replace(tmp,archive)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
        else:
            print(f"Verified existing {dataset} archive.")
        extracted=ds_dir/"extracted"
        extracted.mkdir(exist_ok=True)
        safe_extract(archive,extracted)
        print(f"Verified and extracted {dataset} to {extracted.relative_to(ROOT)}")

if __name__=="__main__": main()
