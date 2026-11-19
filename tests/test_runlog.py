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


def test_sanitize_hides_principal_names_but_keeps_role_name():
    msg = ("User: arn:aws:iam::1:user/example-user is not authorized; "
           "arn:aws:sts::1:assumed-role/Admin/me@example.com; role/bike-demand-predict-role")
    for text in (msg, json.dumps({"e": msg})):
        clean = runlog.sanitize(text, [])
        assert "example-user" not in clean and "me@example.com" not in clean
        assert "role/bike-demand-predict-role" in clean


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
    error = (failed / "error.txt").read_bytes()
    assert b"boom" in error and b"\r\n" not in error


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


def test_interrupt_is_recorded_and_stops(isolated):
    def stop(ctx):
        raise KeyboardInterrupt

    assert run(["stop", "after"], {"stop": stop, "after": lambda ctx: None}) == 130
    r = recs(isolated)
    assert [x["status"] for x in r] == ["failed", "skipped"] and "KeyboardInterrupt" in r[0]["error"]


def test_promote_rolls_back_on_interrupt(isolated, monkeypatch):
    (isolated / "results").mkdir(parents=True, exist_ok=True)
    for name in ("a.txt", "b.txt"):
        (isolated / "results" / name).write_text("old", encoding="utf-8")
    real_copy = runlog.shutil.copy2

    def copy(src, dst):
        if Path(dst).name == "b.txt":
            raise KeyboardInterrupt
        return real_copy(src, dst)
    monkeypatch.setattr(runlog.shutil, "copy2", copy)

    def step(ctx):
        ctx.out("results/a.txt").write_text("new", encoding="utf-8")
        ctx.out("results/b.txt").write_text("new", encoding="utf-8")

    assert run(["step"], {"step": step}) == 130
    for name in ("a.txt", "b.txt"):
        assert (isolated / "results" / name).read_text(encoding="utf-8") == "old"
    assert not list((isolated / "results").glob("*.promote-bak"))


def test_sanitize_ignores_root_home(monkeypatch):
    monkeypatch.setattr(runlog.Path, "home", classmethod(lambda cls: Path("/")))
    assert runlog.sanitize("data/processed/train.csv", []) == "data/processed/train.csv"


def test_load_json_accepts_bom(tmp_path):
    from bikeml.common import load_json
    p = tmp_path / "c.json"
    p.write_bytes(b"\xef\xbb\xbf" + json.dumps({"a": 1}).encode("utf-8"))
    assert load_json(p) == {"a": 1}


def test_sanitize_ignores_path_case():
    raw = str(ROOT / "bikeml" / "runlog.py")
    for text in (raw.lower(), raw.upper(), json.dumps(raw.lower())):
        clean = runlog.sanitize(text, [])
        assert Path.home().name.lower() not in clean.lower()


def test_sanitize_hides_short_repo_path(monkeypatch):
    short = Path(Path.cwd().anchor) / "work"
    monkeypatch.setattr(runlog, "ROOT", short)
    clean = runlog.sanitize(str(short / "config" / "pipeline.json"), [])
    assert clean.startswith("<REPO>") and "work" not in clean


def test_sanitize_hides_account_in_arns_and_profile_names():
    msg = ('User: arn:aws:iam::111122223333:user/example is not authorized on arn:aws:iam::444455556666:role/X; '
           'The profile "mlops" is broken; bucket bike-demand-mlops-ap-northeast-2-1a2b3c4d')
    clean = runlog.sanitize(msg, [("profile", "mlops")])
    assert "111122223333" not in clean and "444455556666" not in clean
    assert 'profile "<PROFILE>"' in clean and "bike-demand-mlops-ap-northeast-2-1a2b3c4d" in clean


def test_common_word_profile_does_not_rewrite_values():
    s = [("profile", "model")]
    assert runlog.sanitize("s3://b/models/v/abc/model.joblib", s) == "s3://b/models/v/abc/model.joblib"
    assert runlog.sanitize("results/cloud/deploy.json", [("profile", "results")]) == "results/cloud/deploy.json"
    assert runlog.sanitize("The config profile (model) could not be found", s) == "The config profile (<PROFILE>) could not be found"
    assert "model" not in runlog.sanitize("Visited profiles: [model, base]", s)


def test_home_path_redaction_stops_at_folder_boundary():
    home = str(Path.home())
    assert runlog.sanitize(home + "\\a.txt", []) == "~\\a.txt"
    assert runlog.sanitize(home + "x\\a.txt", []).startswith(home + "x")


def test_data_files_are_not_rewritten(isolated):
    text = str(Path.home()) + " stays as is"

    def step(ctx):
        ctx.out("data/raw/x.csv").write_text(text, encoding="utf-8")
        ctx.out("results/y.txt").write_text(text, encoding="utf-8")

    assert run(["step"], {"step": step}) == 0
    assert (isolated / "data" / "raw" / "x.csv").read_text(encoding="utf-8") == text
    assert (isolated / "results" / "y.txt").read_text(encoding="utf-8").startswith("~")


def test_odd_run_log_lines_count_as_unreadable(isolated):
    runlog.RUN_LOG.parent.mkdir(parents=True)
    base = {"run_id": "a", "step": "deploy", "status": "success"}
    good = json.dumps({**base, "cloud": {"region": "ap-northeast-2", "bucket": "b"}})
    lines = [good.encode(), b"", b"[]", b"\xff\xfe", json.dumps({**base, "cloud": []}).encode(),
             json.dumps({**base, "cloud": {"region": 5}}).encode(), b'{"run_id": "cut"',
             json.dumps({"cloud": None}).encode(),
             json.dumps({**base, "run_id": "b", "git": "abc", "error": 5, "model": [], "finished_at": 1}).encode()]
    runlog.RUN_LOG.write_bytes(b"\n".join(lines) + b"\n")
    records, broken = runlog.read_run_log()
    assert [r["run_id"] for r in records] == ["a", "b"] and broken == 6
    runlog.print_runs()


def test_runs_with_zero_limit_prints_no_rows(isolated, capsys):
    runlog.RUN_LOG.parent.mkdir(parents=True)
    runlog.RUN_LOG.write_text(json.dumps({"run_id": "a", "step": "s", "status": "success"}) + "\n", encoding="utf-8")
    runlog.print_runs(0)
    assert len(capsys.readouterr().out.strip().splitlines()) == 1


def test_config_path_outside_repo_is_not_recorded(isolated, tmp_path):
    outside = tmp_path / "elsewhere" / "my_config.json"
    args = argparse.Namespace(config=str(outside), lambda_config=str(outside), cmd="run", steps=["ok"])
    assert runlog.run_steps(["ok"], {"ok": lambda ctx: None}, load_config(), args, DEFAULT_CONFIG) == 0
    rec = recs(isolated)[0]
    assert rec["args"]["config"] == "<EXTERNAL>/my_config.json" and "elsewhere" not in json.dumps(rec)


def test_error_text_hides_folder_of_external_config(isolated):
    outside = Path(Path.home().anchor) / "zz_bikeml_external" / "lam.json"
    args = argparse.Namespace(config=str(DEFAULT_CONFIG), lambda_config=str(outside), cmd="run", steps=["bad"])

    def bad(ctx):
        open(outside, encoding="utf-8")
    assert runlog.run_steps(["bad"], {"bad": bad}, load_config(), args, DEFAULT_CONFIG) == 1
    rec = recs(isolated)[0]
    error_txt = (isolated / "results" / "failed" / f"{rec['run_id']}_bad" / "error.txt").read_text(encoding="utf-8")
    assert "<EXTERNAL>" in rec["error"] and "zz_bikeml_external" not in json.dumps(rec)
    assert "zz_bikeml_external" not in error_txt


def test_principal_pattern_only_touches_arns():
    uri = "s3://bike-demand-mlops-x/models/v2-user/abc123def456/model.joblib?versionId=xyz"
    assert runlog.sanitize(uri, []) == uri
    assert runlog.sanitize("arn:aws:iam::1:user/someone", []).endswith(":user/<NAME>")
