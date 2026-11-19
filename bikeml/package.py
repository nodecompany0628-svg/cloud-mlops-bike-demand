"""Lambda 배포 패키지(zip) 만들기.

Windows 에서 만들어도 Lambda(리눅스 x86_64, Python 3.12)에서 돌도록
pip 에 --platform manylinux, --only-binary 를 주어 리눅스용 휠을 받는다.
zip 안의 파일 순서와 시각을 고정해서 같은 입력이면 같은 sha256 이 나온다.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .common import ROOT, load_json, rel, sha256_file, write_json

BUILD = ROOT / "build"
PKG_DIR = BUILD / "lambda_pkg"
ZIP_PATH = BUILD / "lambda_package.zip"
APP_FILES = {
    "lambda_function.py": ROOT / "lambda_src" / "lambda_function.py",
    "bikeml/__init__.py": ROOT / "bikeml" / "__init__.py",
    "bikeml/features.py": ROOT / "bikeml" / "features.py",
    "bikeml/service.py": ROOT / "bikeml" / "service.py",
}
PLATFORMS = ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64"]
UNZIPPED_LIMIT = 250 * 1024 * 1024  # Lambda: 압축 해제 후 250MB 이하
FIXED_TIME = (2020, 1, 1, 0, 0, 0)


def _dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def build_package(lambda_cfg: dict) -> dict:
    py = lambda_cfg["runtime"].replace("python", "")  # "3.12"
    if PKG_DIR.exists():
        shutil.rmtree(PKG_DIR)
    PKG_DIR.mkdir(parents=True)
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", "--no-compile",
           "--target", str(PKG_DIR), "--implementation", "cp", "--python-version", py,
           "--only-binary=:all:", "-r", str(ROOT / "requirements-lambda.txt")]
    for plat in PLATFORMS:
        cmd += ["--platform", plat]
    subprocess.run(cmd, check=True)

    # 실행에 쓰지 않는 테스트 폴더와 캐시는 뺀다(크기 줄이기).
    removed = 0
    for d in sorted(PKG_DIR.rglob("*"), reverse=True):
        if d.is_dir() and d.name in ("tests", "__pycache__"):
            removed += _dir_size(d)
            shutil.rmtree(d)

    for arc, src in APP_FILES.items():
        dst = PKG_DIR / arc
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    unzipped = _dir_size(PKG_DIR)
    if unzipped > UNZIPPED_LIMIT:
        raise RuntimeError(f"압축 해제 크기 {unzipped/1e6:.1f}MB 가 Lambda 한도 250MB 를 넘습니다")

    files = sorted(p for p in PKG_DIR.rglob("*") if p.is_file())
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            info = zipfile.ZipInfo(f.relative_to(PKG_DIR).as_posix(), date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
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
        "platforms": PLATFORMS,
        "installed": installed,
        "app_files": {arc: sha256_file(src) for arc, src in APP_FILES.items()},
    }


def step_package(ctx) -> None:
    lcfg_path = ROOT / (ctx.args.lambda_config or ctx.cfg["cloud"]["lambda_config"])
    ctx.add_config(lcfg_path)
    ctx.add_input(ROOT / "requirements-lambda.txt")
    info = build_package(load_json(lcfg_path))
    write_json(ctx.out("results/cloud/package.json"), info)
    ctx.state["package"] = info
    ctx.summary = {k: info[k] for k in ("zip_sha256", "zip_mb", "unzipped_mb", "file_count")}
