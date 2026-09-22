"""HTTP client with an on-disk cache, retries and honest error classification.

Two failure modes must never be confused (CLAUDE.md rule 7):

* the source *works* but has nothing for this tile -> :class:`DataUnavailableError`
  (or ``None`` when ``allow_missing=True``);
* the source *changed* -- wrong content type, HTML error page, 5xx, moved
  endpoint -> :class:`DataSourceError`, which is never swallowed.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

from rokko_geofusion.config import HttpConfig
from rokko_geofusion.exceptions import DataSourceError, DataUnavailableError
from rokko_geofusion.utils.logging import log_failure_context

logger = logging.getLogger(__name__)

#: Content types that indicate an error page rather than the payload we asked for.
_ERROR_CONTENT_HINTS = ("text/html", "application/xml", "text/xml")


def _cache_key(url: str, params: Mapping[str, Any] | None, body: str | None) -> str:
    material = url
    if params:
        material += "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    if body:
        material += "#" + hashlib.sha256(body.encode("utf-8")).hexdigest()
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class HttpClient:
    """Small, dependency-light client tuned for tile servers and Overpass.

    ``cache_dir`` stores raw response bodies keyed by URL so that re-running a
    stage does not re-download anything -- important both for reproducibility
    (V6) and for not hammering public services.
    """

    def __init__(
        self,
        config: HttpConfig,
        *,
        cache_dir: Path | None = None,
        allow_network: bool = True,
        user_agent: str | None = None,
    ) -> None:
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.allow_network = allow_network
        self.user_agent = user_agent or config.user_agent
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        self.stats = {"hits": 0, "misses": 0, "missing": 0, "errors": 0}
        if self.cache_dir and config.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache ---------------------------------------------------------------
    def _cache_path(self, key: str) -> Path | None:
        if not (self.cache_dir and self.config.cache_enabled):
            return None
        return self.cache_dir / key[:2] / f"{key}.bin"

    def _read_cache(self, key: str) -> bytes | None:
        path = self._cache_path(key)
        if path and path.is_file():
            self.stats["hits"] += 1
            return path.read_bytes()
        return None

    def _write_cache(self, key: str, url: str, payload: bytes) -> None:
        path = self._cache_path(key)
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        path.with_suffix(".url").write_text(url, encoding="utf-8")

    # -- request -------------------------------------------------------------
    def _throttle(self) -> None:
        interval = self.config.min_interval_s
        if interval <= 0:
            return
        with self._lock:
            wait = self._last_request_at + interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request_at = time.monotonic()

    def _looks_like_error_page(self, response: requests.Response, expect: str | None) -> bool:
        if expect is None:
            return False
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if not content_type:
            return False
        if content_type.startswith(expect):
            return False
        return any(content_type.startswith(hint) for hint in _ERROR_CONTENT_HINTS)

    def request(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Mapping[str, Any] | None = None,
        data: str | None = None,
        expect_content_type: str | None = None,
        allow_missing: bool = False,
        force_refresh: bool = False,
        backoff_s: float | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> bytes | None:
        """Fetch ``url``; return the body, or ``None`` when legitimately absent.

        ``backoff_s`` overrides the configured retry delay for services with
        their own pacing rules (Overpass hands out slots on a ~60 s cycle).
        """
        backoff = self.config.backoff_s if backoff_s is None else backoff_s
        key = _cache_key(url, params, data)
        if not force_refresh:
            cached = self._read_cache(key)
            if cached is not None:
                return cached

        if not self.allow_network:
            raise DataSourceError(
                f"network access is disabled (runtime.allow_network=false) but {url} "
                "is not in the local cache"
            )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            self._throttle()
            try:
                response = self._session.request(
                    method,
                    url,
                    params=dict(params) if params else None,
                    data=data,
                    timeout=self.config.timeout_s,
                )
            except requests.RequestException as exc:
                last_error = exc
                logger.debug("attempt %d/%d failed for %s: %s",
                             attempt, self.config.max_retries, url, exc)
                if attempt < self.config.max_retries:
                    time.sleep(backoff * attempt)
                continue

            if response.status_code == 404:
                self.stats["missing"] += 1
                if allow_missing:
                    logger.debug("no data at %s (HTTP 404)", url)
                    return None
                raise DataUnavailableError(f"no data available at {url} (HTTP 404)")

            if response.status_code in (429, 503, 504):
                last_error = DataSourceError(
                    f"{url} returned HTTP {response.status_code} (rate limited / busy)"
                )
                wait = backoff * attempt * 2
                logger.warning("HTTP %d from %s; retrying in %.1fs (attempt %d/%d)",
                               response.status_code, url, wait, attempt, self.config.max_retries)
                time.sleep(wait)
                continue

            if not response.ok:
                self.stats["errors"] += 1
                log_failure_context(
                    logger,
                    what=f"HTTP {response.status_code} from data source",
                    url=url,
                    cause="the endpoint may have moved or changed its request format",
                    **(context or {}),
                )
                raise DataSourceError(
                    f"{url} returned HTTP {response.status_code}: "
                    f"{response.text[:200]!r}"
                )

            if self._looks_like_error_page(response, expect_content_type):
                self.stats["errors"] += 1
                content_type = response.headers.get("Content-Type", "?")
                log_failure_context(
                    logger,
                    what="unexpected response content type",
                    url=url,
                    expected=expect_content_type,
                    received=content_type,
                    cause="the upstream API changed, or the request was rejected",
                    **(context or {}),
                )
                raise DataSourceError(
                    f"{url} returned Content-Type {content_type!r}, expected "
                    f"{expect_content_type!r}. Treating this as an upstream API change, "
                    "not as missing data."
                )

            payload = response.content
            self.stats["misses"] += 1
            self._write_cache(key, url, payload)
            return payload

        self.stats["errors"] += 1
        log_failure_context(
            logger,
            what="exhausted retries for data source",
            url=url,
            attempts=self.config.max_retries,
            cause=str(last_error) if last_error else "unknown",
            **(context or {}),
        )
        raise DataSourceError(
            f"could not fetch {url} after {self.config.max_retries} attempts: {last_error}"
        )

    def get_many(
        self,
        urls: Sequence[str],
        *,
        max_workers: int | None = None,
        expect_content_type: str | None = None,
        allow_missing: bool = True,
        force_refresh: bool = False,
        progress: bool = True,
        desc: str = "downloading",
    ) -> dict[str, bytes | None]:
        """Fetch many URLs concurrently, preserving per-URL failure semantics."""
        workers = max(1, min(max_workers or self.config.max_workers, len(urls) or 1))
        results: dict[str, bytes | None] = {}

        def fetch(url: str) -> tuple[str, bytes | None]:
            return url, self.request(
                url,
                expect_content_type=expect_content_type,
                allow_missing=allow_missing,
                force_refresh=force_refresh,
            )

        iterator: Iterable[tuple[str, bytes | None]]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(fetch, urls)
            if progress:
                iterator = _maybe_progress(iterator, total=len(urls), desc=desc)
            for url, payload in iterator:
                results[url] = payload
        return results

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _maybe_progress(iterable: Iterable[Any], *, total: int, desc: str) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="req", leave=False)


def client_from_config(config: Any, *, cache_subdir: str | None = None) -> HttpClient:
    """Build a client from a :class:`~rokko_geofusion.config.Config`."""
    cache_dir = config.paths.cache
    if cache_subdir:
        cache_dir = cache_dir / cache_subdir
    return HttpClient(
        config.processing.http,
        cache_dir=cache_dir,
        allow_network=config.runtime.allow_network,
    )
