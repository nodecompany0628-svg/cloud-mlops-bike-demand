from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "pipeline.json"


def load_json(path) -> dict:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_config(path=None) -> dict:
    return load_json(path or DEFAULT_CONFIG)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def model_dir(cfg: dict) -> Path:
    return ROOT / "models" / cfg["model"]["version"]


def candidate_dir(cfg: dict) -> Path:
    return ROOT / "models" / "_candidate" / cfg["model"]["version"]


def resolve_profile(cfg: dict, cli_profile: str | None) -> str | None:
    return cli_profile or os.environ.get("AWS_PROFILE") or cfg["cloud"].get("profile")


def resolve_region(cfg: dict, cli_region: str | None) -> str:
    return cli_region or cfg["cloud"]["region"]
