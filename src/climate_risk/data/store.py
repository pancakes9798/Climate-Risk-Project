"""Download the public sources and build the small, versioned extracts used by the models.

``data/raw`` holds the files exactly as published (git-ignored). ``data/processed`` holds what
the models read: month-end GSW curves and the monthly endpoint drivers. Committing the
processed extracts pins the data vintage behind the published results; ``climate-risk download``
refreshes them.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from climate_risk.config import Paths, TermStructureConfig
from climate_risk.data import sources

LOGGER = logging.getLogger(__name__)
USER_AGENT = "Mozilla/5.0 (compatible; climate-risk research pipeline)"

RAW_FILES = {
    "gsw": ("feds200628.csv", sources.GSW_URL),
    "hlw": ("Holston_Laubach_Williams_current_estimates.xlsx", sources.HLW_URL),
    "expinf": ("EXPINF10YR.csv", sources.FRED_URL.format(series_id="EXPINF10YR")),
    "baa": ("BAA10Y.csv", sources.FRED_URL.format(series_id="BAA10Y")),
}
HTTPS_ONLY = frozenset({"https"})
CURVES_FILE = "gsw_zero_curves_monthly.csv"
ENDPOINT_FILE = "endpoint_drivers_monthly.csv"
HLW_FILE = "hlw_us_quarterly.csv"


class IncompleteDownloadError(OSError):
    """The server closed the connection before sending the announced number of bytes."""


def _announced_length(response: Any) -> int | None:
    """Content-Length of a non-chunked response, when the server sent a valid one."""
    headers = getattr(response, "headers", None) or {}
    if str(headers.get("Transfer-Encoding", "")).lower() == "chunked":
        return None  # http.client ignores Content-Length on chunked bodies
    try:
        return int(headers["Content-Length"])
    except (KeyError, TypeError, ValueError):
        return None


def download(
    url: str,
    destination: Path,
    timeout: float = 120.0,
    *,
    schemes: frozenset[str] = HTTPS_ONLY,
) -> Path:
    """Fetch ``url`` into ``destination`` atomically; only the URL ``schemes`` are accepted.

    The body is written to a temporary file that replaces ``destination`` only once it is
    complete: it must not be empty and, when the server announces a Content-Length, the number
    of bytes received must match it. A failed or truncated download leaves the previous file
    untouched.
    """
    scheme = urllib.parse.urlsplit(url).scheme
    if scheme not in schemes:
        raise ValueError(f"refusing to fetch {url!r}: the scheme must be one of {sorted(schemes)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_path: Path | None = None
    try:
        # The scheme is validated above, so urlopen cannot read local files (bandit B310).
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,  # nosec B310
            tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp,
        ):
            tmp_path = Path(tmp.name)
            shutil.copyfileobj(response, tmp)
            expected = _announced_length(response)
            received = tmp.tell()
        if received == 0:
            raise IncompleteDownloadError(f"{url}: empty response")
        if expected is not None and received != expected:
            raise IncompleteDownloadError(
                f"{url}: received {received} of {expected} announced bytes"
            )
        tmp_path.chmod(0o644)
        tmp_path.replace(destination)
    except BaseException:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    LOGGER.info("downloaded %s -> %s", url, destination)
    return destination


def refresh_raw(paths: Paths) -> dict[str, Path]:
    """Download every public source into ``data/raw``."""
    return {key: download(url, paths.raw / name) for key, (name, url) in RAW_FILES.items()}


def raw_path(paths: Paths, key: str) -> Path:
    path = paths.raw / RAW_FILES[key][0]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `climate-risk download` first")
    return path


def build_processed(paths: Paths, cfg: TermStructureConfig) -> dict[str, Path]:
    """Write the month-end curves, the HLW estimates and the endpoint drivers."""
    paths.processed.mkdir(parents=True, exist_ok=True)
    curves = sources.month_end_curves(
        sources.read_gsw_zero_yields(raw_path(paths, "gsw")), cfg.maturities
    )
    hlw = sources.read_hlw_us(raw_path(paths, "hlw"))
    inflation = sources.to_monthly(sources.read_fred_csv(raw_path(paths, "expinf")))
    rstar = sources.quarterly_to_monthly(hlw, cfg.endpoint_lag_quarters)["rstar"]
    drivers = pd.DataFrame({"rstar": rstar, "inflation": inflation}).dropna()
    drivers["istar"] = drivers["rstar"] + drivers["inflation"]

    out = {
        "curves": paths.processed / CURVES_FILE,
        "hlw": paths.processed / HLW_FILE,
        "endpoint": paths.processed / ENDPOINT_FILE,
    }
    curves.round(6).to_csv(out["curves"])
    hlw.round(6).to_csv(out["hlw"])
    drivers.round(6).to_csv(out["endpoint"])
    return out


def _read_monthly(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M", name="month")
    return frame


def load_curves(paths: Paths) -> pd.DataFrame:
    """Month-end zero-coupon curves; columns are maturities in years (floats)."""
    curves = _read_monthly(paths.processed / CURVES_FILE)
    curves.columns = [float(c) for c in curves.columns]
    return curves


def load_endpoint_drivers(paths: Paths) -> pd.DataFrame:
    """Monthly r* (lagged one quarter), 10y expected inflation and i* = r* + π*."""
    return _read_monthly(paths.processed / ENDPOINT_FILE)


def load_baa_spread(paths: Paths) -> pd.Series:
    """Moody's Baa – 10y Treasury spread (percent), monthly average; demo-mode credit input."""
    return sources.to_monthly(sources.read_fred_csv(raw_path(paths, "baa")), how="mean")
