import argparse
import json
from pathlib import Path

import pytest

from bikeml import runlog
from bikeml.common import DEFAULT_CONFIG, ROOT, load_config


def test_sanitize_hides_paths_at_every_escape_level():
    raw = str(ROOT / "models" / "x.joblib")
    once = json.dumps(raw)
    twice = json.dumps(json.dumps({"m": raw}))
    for text in (raw, once, twice, (ROOT / "a").as_posix()):
        clean = runlog.sanitize(text, [])
        assert str(ROOT) not in clean and str(ROOT).replace("\\", "\\\\") not in clean
        assert Path.home().name not in clean


def test_sanitize_hides_account_inside_tuples_and_lists():
    obj = ({"arn": "arn:aws:iam::111122223333:role/x"}, ["111122223333"])
    clean = runlog.sanitize(obj, ["111122223333"])
    assert isinstance(clean, tuple) and "111122223333" not in json.dumps(clean)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "ROOT", tmp_path)
    monkeypatch.setattr(runlog, "RUN_LOG", tmp_path / "runs" / "run_log.jsonl")
    monkeypatch.setattr(runlog, "STAGING", tmp_path / "results" / "_staging")
    monkeypatch.setattr(runlog, "FAILED", tmp_path / "results" / "failed")
    return tmp_path


def run(steps, registry):
    args = argparse.Namespace(config=str(DEFAULT_CONFIG), cmd="run", steps=steps)
    return runlog.run_steps(steps, registry, load_config(), args, DEFAULT_CONFIG)


def recs(root):
    return [json.loads(x) for x in (root / "runs" / "run_log.jsonl").read_text(encoding="utf-8").splitlines()]


def test_failure_stops_later_steps_and_keeps_good_results(isolated):
    (isolated / "results").mkdir(parents=True, exist_ok=True)
    (isolated / "results" / "a.txt").write_text("old", encoding="utf-8")
    called = []

    def ok(ctx):
        ctx.out("results/a.txt").write_text("new", encoding="utf-8")

    def bad(ctx):
        ctx.out("results/a.txt").write_text("broken", encoding="utf-8")
        raise ValueError("boom")

    def never(ctx):
        called.append(1)

    assert run(["ok", "bad", "never"], {"ok": ok, "bad": bad, "never": never}) == 1
    r = recs(isolated)
    assert [x["status"] for x in r] == ["success", "failed", "skipped"] and not called
    assert (isolated / "results" / "a.txt").read_text(encoding="utf-8") == "new"
    failed = isolated / "results" / "failed" / f"{r[1]['run_id']}_bad"
    assert (failed / "results" / "a.txt").read_text(encoding="utf-8") == "broken"
    assert "boom" in (failed / "error.txt").read_text(encoding="utf-8")


def test_promote_is_all_or_nothing(isolated, monkeypatch):
    (isolated / "results").mkdir(parents=True, exist_ok=True)
    for name in ("a.txt", "b.txt"):
        (isolated / "results" / name).write_text("old", encoding="utf-8")
    real_copy = runlog.shutil.copy2

    def copy(src, dst):
        if Path(dst).name == "b.txt":
            raise PermissionError("locked")
        return real_copy(src, dst)
    monkeypatch.setattr(runlog.shutil, "copy2", copy)

    def step(ctx):
        ctx.out("results/a.txt").write_text("new", encoding="utf-8")
        ctx.out("results/b.txt").write_text("new", encoding="utf-8")

    assert run(["step"], {"step": step}) == 1
    for name in ("a.txt", "b.txt"):
        assert (isolated / "results" / name).read_text(encoding="utf-8") == "old"
    assert not list((isolated / "results").glob("*.promote-bak"))


def test_non_utf8_output_fails_the_step(isolated):
    def step(ctx):
        ctx.out("results/x.csv").write_bytes("가".encode("cp949"))

    assert run(["step"], {"step": step}) == 1
    r = recs(isolated)
    assert "UTF-8" in r[0]["error"]
    assert not (isolated / "results" / "x.csv").exists()
    assert not (isolated / "results" / "failed" / f"{r[0]['run_id']}_step" / "results" / "x.csv").exists()
