"""HTTP client: caching, retries and error classification."""

from __future__ import annotations

import pytest
import requests

from rokko_geofusion.config import HttpConfig
from rokko_geofusion.exceptions import DataSourceError, DataUnavailableError
from rokko_geofusion.io.http import HttpClient


class _FakeResponse:
    def __init__(self, status_code=200, content=b"ok", content_type="text/plain"):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.text = content.decode("utf-8", "replace")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class _FakeSession:
    """Replays a scripted sequence of responses/exceptions and counts calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


@pytest.fixture()
def http_config():
    return HttpConfig(timeout_s=1.0, max_retries=3, backoff_s=0.0, cache_enabled=True)


def _client(http_config, tmp_path, script):
    client = HttpClient(http_config, cache_dir=tmp_path / "cache")
    client._session = _FakeSession(script)
    return client


def test_successful_request_is_cached(http_config, tmp_path):
    client = _client(http_config, tmp_path, [_FakeResponse(content=b"payload")])
    assert client.request("https://example.org/a") == b"payload"
    assert len(client._session.calls) == 1
    # Second call must be served from disk, not from the network.
    assert client.request("https://example.org/a") == b"payload"
    assert len(client._session.calls) == 1
    assert client.stats["hits"] == 1


def test_force_refresh_bypasses_the_cache(http_config, tmp_path):
    client = _client(http_config, tmp_path,
                     [_FakeResponse(content=b"v1"), _FakeResponse(content=b"v2")])
    assert client.request("https://example.org/a") == b"v1"
    assert client.request("https://example.org/a", force_refresh=True) == b"v2"


def test_cache_key_separates_urls(http_config, tmp_path):
    client = _client(http_config, tmp_path, [_FakeResponse(content=b"one")])
    client.request("https://example.org/a")
    client._session.script = [_FakeResponse(content=b"two")]
    assert client.request("https://example.org/b") == b"two"


def test_404_is_missing_data_not_a_source_error(http_config, tmp_path):
    client = _client(http_config, tmp_path, [_FakeResponse(status_code=404)])
    assert client.request("https://example.org/gone", allow_missing=True) is None
    assert client.stats["missing"] == 1


def test_404_raises_when_not_allowed(http_config, tmp_path):
    client = _client(http_config, tmp_path, [_FakeResponse(status_code=404)])
    with pytest.raises(DataUnavailableError):
        client.request("https://example.org/gone")


def test_500_is_a_source_error(http_config, tmp_path):
    client = _client(http_config, tmp_path, [_FakeResponse(status_code=500, content=b"boom")])
    with pytest.raises(DataSourceError):
        client.request("https://example.org/broken", allow_missing=True)


def test_html_error_page_is_not_mistaken_for_data(http_config, tmp_path):
    """An upstream API change must never look like 'no data for this tile'."""
    client = _client(
        http_config,
        tmp_path,
        [_FakeResponse(content=b"<html>moved</html>", content_type="text/html")],
    )
    with pytest.raises(DataSourceError, match="upstream API change"):
        client.request(
            "https://example.org/tile.txt",
            expect_content_type="text/plain",
            allow_missing=True,
        )


def test_matching_content_type_is_accepted(http_config, tmp_path):
    client = _client(http_config, tmp_path,
                     [_FakeResponse(content=b"1,2", content_type="text/plain; charset=utf-8")])
    assert client.request("https://e.org/t", expect_content_type="text/plain") == b"1,2"


def test_connection_errors_are_retried_then_reported(http_config, tmp_path):
    client = _client(http_config, tmp_path, [requests.ConnectionError("no route")])
    with pytest.raises(DataSourceError, match="after 3 attempts"):
        client.request("https://example.org/down")
    assert len(client._session.calls) == 3


def test_transient_failure_then_success(http_config, tmp_path):
    client = _client(
        http_config, tmp_path,
        [requests.ConnectionError("flaky"), _FakeResponse(content=b"late")],
    )
    assert client.request("https://example.org/flaky") == b"late"


def test_rate_limit_is_retried(http_config, tmp_path):
    client = _client(http_config, tmp_path,
                     [_FakeResponse(status_code=429), _FakeResponse(content=b"ok")])
    assert client.request("https://example.org/busy") == b"ok"
    assert len(client._session.calls) == 2


def test_offline_mode_refuses_uncached_urls(http_config, tmp_path):
    client = HttpClient(http_config, cache_dir=tmp_path / "c", allow_network=False)
    with pytest.raises(DataSourceError, match="network access is disabled"):
        client.request("https://example.org/a")


def test_offline_mode_still_serves_the_cache(http_config, tmp_path):
    warm = _client(http_config, tmp_path, [_FakeResponse(content=b"cached")])
    warm.request("https://example.org/a")
    offline = HttpClient(http_config, cache_dir=tmp_path / "cache", allow_network=False)
    assert offline.request("https://example.org/a") == b"cached"


def test_get_many_preserves_per_url_results(http_config, tmp_path):
    client = _client(http_config, tmp_path, [_FakeResponse(content=b"x")])
    urls = [f"https://example.org/{i}" for i in range(5)]
    results = client.get_many(urls, progress=False)
    assert set(results) == set(urls)
    assert all(value == b"x" for value in results.values())
