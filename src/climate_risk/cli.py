"""Command-line interface: ``climate-risk {download,run,info}``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from climate_risk.config import Config, Paths

LOGGER = logging.getLogger("climate_risk")


def _config(root: Path | None) -> Config:
    return Config() if root is None else replace(Config(), paths=Paths(root=root))


def _download(cfg: Config) -> int:
    from climate_risk.data import store

    store.refresh_raw(cfg.paths)
    for name, path in store.build_processed(cfg.paths, cfg.term_structure).items():
        LOGGER.info("processed %s -> %s", name, path)
    return 0


def _run(cfg: Config, *, backtest: bool, figures: bool) -> int:
    from climate_risk import report
    from climate_risk.pipeline import run_all, write_tables

    results = run_all(cfg, with_backtest=backtest)
    write_tables(results.tables, cfg.paths.tables)
    report.write(results, cfg.paths.results / "README.md")
    if figures:
        from climate_risk.figures import render_all

        render_all(results, cfg.paths.figures)
    LOGGER.info("credit inputs: %s mode", results.credit.mode.upper())
    return 0


def _info(cfg: Config) -> int:
    paths = cfg.paths
    rows = {
        "repository root": paths.root,
        "processed data": paths.processed,
        "scenario table": paths.scenarios_csv,
        "CDS panel": paths.cds_csv,
        "credit mode": "cds (licensed panel found)" if paths.cds_csv.exists() else "demo",
        "jump-off month": cfg.term_structure.as_of,
    }
    width = max(map(len, rows))
    for key, value in rows.items():
        print(f"{key:<{width}}  {value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="climate-risk",
        description="Scenario-conditional yield curves and climate spreads to 2100.",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="repository/data root (default: this repository)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="fetch public data and rebuild data/processed")
    run = sub.add_parser("run", help="run the full pipeline and write results/")
    run.add_argument("--no-backtest", action="store_true", help="skip the forecast backtest")
    run.add_argument("--no-figures", action="store_true", help="tables only")
    sub.add_parser("info", help="show paths and the credit-data mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = _config(args.root)
    if args.command == "download":
        return _download(cfg)
    if args.command == "run":
        return _run(cfg, backtest=not args.no_backtest, figures=not args.no_figures)
    return _info(cfg)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
