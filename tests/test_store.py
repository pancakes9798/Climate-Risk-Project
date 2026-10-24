import io
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from climate_risk.config import Paths, TermStructureConfig
from climate_risk.data import store

GSW = """\
"Note: staff research product."

Date,BETA0,SVENY01,SVENY10
2020-01-31,1,1.50,1.90
2020-02-28,1,1.40,1.70
2020-03-31,1,1.30,1.60
2020-04-30,1,1.20,1.50
"""


def _write_raw(paths: Paths) -> None:
    paths.raw.mkdir(parents=True)
    (paths.raw / store.RAW_FILES["gsw"][0]).write_text(GSW)
    (paths.raw / store.RAW_FILES["expinf"][0]).write_text(
        "observation_date,EXPINF10YR\n2020-01-01,2.0\n2020-02-01,2.1\n2020-03-01,2.2\n"
        "2020-04-01,2.3\n"
    )
    rows = [
        [None, None, "Trend Growth (g)", None, "Other Determinants (z)", None, "Natural Rate (r*)"],
        ["Date", None, "US", "CA", "US", "CA", "US"],
        [pd.Timestamp("2019-10-01"), None, 1.8, 1.0, -1.0, 0.0, 0.5],
        [pd.Timestamp("2020-01-01"), None, 1.7, 1.0, -1.0, 0.0, 0.7],
    ]
    with pd.ExcelWriter(paths.raw / store.RAW_FILES["hlw"][0]) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="HLW Estimates", header=False, index=False)


def test_build_and_load_processed_roundtrip(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    _write_raw(paths)
    cfg = TermStructureConfig(maturities=(1.0, 10.0), endpoint_lag_quarters=1)
    written = store.build_processed(paths, cfg)
    assert all(p.exists() for p in written.values())
    curves = store.load_curves(paths)
    assert list(curves.columns) == [1.0, 10.0]
    assert curves.index.freqstr == "M"
    drivers = store.load_endpoint_drivers(paths)
    # Q4-2019 r* (0.5) is used in Q1-2020 months, Q1-2020 r* (0.7) from April
    assert drivers.loc[pd.Period("2020-02", "M"), "istar"] == pytest.approx(0.5 + 2.1)
    assert drivers.loc[pd.Period("2020-04", "M"), "rstar"] == pytest.approx(0.7)


def test_missing_raw_file_has_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="climate-risk download"):
        store.raw_path(Paths(root=tmp_path), "gsw")


def test_failed_download_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload")
    target = tmp_path / "out" / "copy.txt"

    def broken_copy(*_: object) -> None:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(store.shutil, "copyfileobj", broken_copy)
    with pytest.raises(RuntimeError, match="connection reset"):
        store.download(source.as_uri(), target, schemes=frozenset({"file"}))
    assert list(target.parent.iterdir()) == []


class _TruncatedResponse(io.BytesIO):
    headers: ClassVar[dict[str, str]] = {"Content-Length": "5000"}

    def __enter__(self) -> "_TruncatedResponse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _ChunkedResponse(_TruncatedResponse):
    headers: ClassVar[dict[str, str]] = {"Content-Length": "5000", "Transfer-Encoding": "chunked"}


class _GarbledLengthResponse(_TruncatedResponse):
    headers: ClassVar[dict[str, str]] = {"Content-Length": "n/a"}


@pytest.mark.parametrize("response", [_ChunkedResponse, _GarbledLengthResponse])
def test_length_check_skips_chunked_or_unparsable_lengths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, response: type[io.BytesIO]
) -> None:
    monkeypatch.setattr(store.urllib.request, "urlopen", lambda *_, **__: response(b"complete"))
    target = store.download("https://example.org/data.csv", tmp_path / "data.csv")
    assert target.read_bytes() == b"complete"


def test_empty_download_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.csv"
    target.write_text("previous complete file")
    monkeypatch.setattr(store.urllib.request, "urlopen", lambda *_, **__: _ChunkedResponse(b""))
    with pytest.raises(store.IncompleteDownloadError, match="empty"):
        store.download("https://example.org/data.csv", target)
    assert target.read_text() == "previous complete file"


def test_truncated_download_keeps_the_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.csv"
    target.write_text("previous complete file")
    monkeypatch.setattr(
        store.urllib.request, "urlopen", lambda *_, **__: _TruncatedResponse(b"39 bytes only")
    )
    with pytest.raises(store.IncompleteDownloadError, match="announced"):
        store.download("https://example.org/data.csv", target)
    assert target.read_text() == "previous complete file"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.csv"]


def test_download_writes_atomically(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload")
    target = store.download(
        source.as_uri(), tmp_path / "out" / "copy.txt", schemes=frozenset({"file"})
    )
    assert target.read_text() == "payload"
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("url", ["file:///etc/hosts", "http://example.org/data.csv", "data.csv"])
def test_only_https_sources_are_fetched_by_default(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="scheme must be one of"):
        store.download(url, tmp_path / "data.csv")
    assert not (tmp_path / "data.csv").exists()


def test_public_sources_use_https() -> None:
    assert all(url.startswith("https://") for _, url in store.RAW_FILES.values())
