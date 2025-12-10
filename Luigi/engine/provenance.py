import os, time, json, hashlib
from typing import Iterable

def sha256_file(fp: str) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_sidecar(outputs: Iterable, params: dict) -> None:
    outs = list(outputs)
    if not outs: return
    prov_path = os.path.join(os.path.dirname(outs[0].path), "_provenance.json")
    meta = {
        "params": params,
        "tool": {"name": "Radiuma-Luigi", "version": params.get("tool_version", "0.1.0")},
        "timestamps": {"finished_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        "checksums": {}
    }
    for o in outs:
        p = o.path
        if os.path.exists(p):
            meta["checksums"][os.path.basename(p)] = sha256_file(p)
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
