from __future__ import annotations

import importlib.util
import py_compile
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .common import ROOT, load_json, rel, sha256_file, write_json

BUILD = ROOT / "build"
PKG_DIR = BUILD / "lambda_pkg"
ZIP_PATH = BUILD / "lambda_package.zip"
REQUIREMENTS = ROOT / "requirements-lambda.txt"
APP_FILES = {
    "lambda_function.py": ROOT / "lambda_src" / "lambda_function.py",
    "bikeml/__init__.py": ROOT / "bikeml" / "__init__.py",
    "bikeml/features.py": ROOT / "bikeml" / "features.py",
    "bikeml/service.py": ROOT / "bikeml" / "service.py",
}
PLATFORMS = {
    "x86_64": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"],
    "arm64": ["manylinux_2_28_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64"],
}
UNZIPPED_LIMIT = 250 * 1024 * 1024
FIXED_TIME = (2020, 1, 1, 0, 0, 0)


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def build_package(lambda_cfg: dict) -> dict:
    py = lambda_cfg["runtime"].replace("python", "")
    if PKG_DIR.exists():
        shutil.rmtree(PKG_DIR)
    PKG_DIR.mkdir(parents=True)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-compile",
           "--target", str(PKG_DIR), "--implementation", "cp", "--python-version", py,
           "--only-binary=:all:", "-r", str(REQUIREMENTS)]
    platforms = PLATFORMS[lambda_cfg["architecture"]]
    for plat in platforms:
        cmd += ["--platform", plat]
    subprocess.run(cmd, check=True)

    removed = _dir_size(PKG_DIR / "bin") if (PKG_DIR / "bin").exists() else 0
    shutil.rmtree(PKG_DIR / "bin", ignore_errors=True)
    for record in PKG_DIR.glob("*.dist-info/RECORD"):
        lines = record.read_text(encoding="utf-8").splitlines(keepends=True)
        record.write_text("".join(x for x in lines if not x.startswith("../../bin/")), encoding="utf-8", newline="")
    for d in sorted(PKG_DIR.rglob("*"), reverse=True):
        if d.is_dir() and d.name in ("tests", "__pycache__"):
            removed += _dir_size(d)
            shutil.rmtree(d)

    for arc, src in APP_FILES.items():
        dst = PKG_DIR / arc
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    precompiled = f"{sys.version_info.major}.{sys.version_info.minor}" == py
    if precompiled:
        for src in sorted(PKG_DIR.rglob("*.py")):
            try:
                py_compile.compile(str(src), cfile=importlib.util.cache_from_source(str(src)),
                                   dfile="/var/task/" + src.relative_to(PKG_DIR).as_posix(), doraise=True,
                                   invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
            except py_compile.PyCompileError:
                pass

    unzipped = _dir_size(PKG_DIR)
    if unzipped > UNZIPPED_LIMIT:
        raise RuntimeError(f"압축 해제 크기 {unzipped / 1e6:.1f}MB 가 Lambda 한도 250MB 를 넘습니다")

    files = sorted(p for p in PKG_DIR.rglob("*") if p.is_file())
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            info = zipfile.ZipInfo(f.relative_to(PKG_DIR).as_posix(), date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, f.read_bytes())

    installed = sorted(p.name for p in PKG_DIR.glob("*.dist-info"))
    return {
        "zip": rel(ZIP_PATH),
        "zip_sha256": sha256_file(ZIP_PATH),
        "zip_mb": round(ZIP_PATH.stat().st_size / 1e6, 2),
        "unzipped_mb": round(unzipped / 1e6, 2),
        "removed_test_dirs_mb": round(removed / 1e6, 2),
        "file_count": len(files),
        "runtime": lambda_cfg["runtime"],
        "architecture": lambda_cfg["architecture"],
        "precompiled_pyc": precompiled,
        "platforms": platforms,
        "installed": installed,
        "app_files": {arc: sha256_file(src) for arc, src in APP_FILES.items()},
        "requirements_sha256": sha256_file(REQUIREMENTS),
    }


def step_package(ctx) -> None:
    lcfg_path = ROOT / (ctx.args.lambda_config or ctx.cfg["cloud"]["lambda_config"])
    ctx.add_config(lcfg_path)
    ctx.add_input(REQUIREMENTS)
    info = build_package(load_json(lcfg_path))
    write_json(ctx.out("results/cloud/package.json"), info)
    ctx.state["package"] = info
    ctx.summary = {k: info[k] for k in ("zip_sha256", "zip_mb", "unzipped_mb", "file_count")}
