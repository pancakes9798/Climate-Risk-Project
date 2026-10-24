"""Command-line entry points on the synthetic data root."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from climate_risk import cli
from climate_risk.config import Config, Paths
from climate_risk.data import store
from tests.conftest import small_config

pytestmark = pytest.mark.usefixtures("no_cds_override")


def test_info_reports_paths_and_mode(data_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--root", str(data_root), "info"]) == 0
    out = capsys.readouterr().out
    assert "cds (licensed panel found)" in out
    assert str(data_root) in out


def test_parser_flags() -> None:
    args = cli.build_parser().parse_args(["run", "--no-backtest", "--no-figures"])
    assert args.no_backtest
    assert args.no_figures


def test_run_writes_tables_and_summary(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    shutil.copytree(data_root, root)
    monkeypatch.setattr(cli, "_config", lambda path: small_config(path))
    assert cli.main(["--root", str(root), "run", "--no-backtest", "--no-figures"]) == 0
    assert (root / "results" / "README.md").read_text(encoding="utf-8").startswith("# Results")
    assert (root / "results" / "tables" / "climate_spreads.csv").exists()
    assert not (root / "results" / "figures").exists()


def test_download_refreshes_raw_data_then_rebuilds_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []

    def refresh(paths: Paths) -> dict[str, Path]:
        calls.append(("raw", paths.root))
        return {}

    def build(paths: Paths, _: object) -> dict[str, Path]:
        calls.append(("processed", paths.root))
        return {"curves": paths.processed / store.CURVES_FILE}

    monkeypatch.setattr(store, "refresh_raw", refresh)
    monkeypatch.setattr(store, "build_processed", build)
    assert cli.main(["--root", str(tmp_path), "download"]) == 0
    assert calls == [("raw", tmp_path), ("processed", tmp_path)]


def test_default_configuration_uses_the_repository_root() -> None:
    assert cli._config(None) == Config()
