"""Minimal CKAN catalogue client used to *discover* -- never to guess -- data.

The project brief names the Hyogo prefecture high-accuracy 3D dataset as the
preferred LiDAR source, but no stable public download URL for it could be
verified. Instead of inventing one, this module lets the user search a CKAN
catalogue (the G-Spatial Information Center runs one) and pin the resource URL
they choose into ``configs/*.yaml``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from rokko_geofusion.exceptions import DataSourceError
from rokko_geofusion.io.http import HttpClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CkanResource:
    name: str
    url: str
    format: str
    size: int | None = None

    def __str__(self) -> str:
        size = f"  {self.size / 1e6:.1f} MB" if self.size else ""
        return f"[{self.format or '?':>8}] {self.name}{size}\n           {self.url}"


@dataclass(frozen=True)
class CkanDataset:
    id: str
    title: str
    organization: str | None
    resources: tuple[CkanResource, ...]

    def __str__(self) -> str:
        head = f"{self.title}  (id={self.id}, org={self.organization or '-'})"
        body = "\n".join(f"    {resource}" for resource in self.resources)
        return f"{head}\n{body}" if body else head


def _api(client: HttpClient, catalog_url: str, action: str,
         params: dict[str, Any]) -> dict[str, Any]:
    url = f"{catalog_url.rstrip('/')}/api/3/action/{action}"
    payload = client.request(url, params=params, expect_content_type="application/json")
    if payload is None:
        raise DataSourceError(f"CKAN action {action} returned no payload ({url})")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataSourceError(
            f"CKAN action {action} at {url} did not return JSON; the catalogue API "
            f"may have changed: {exc}"
        ) from exc
    if not document.get("success", False):
        raise DataSourceError(f"CKAN action {action} failed: {document.get('error')}")
    return document["result"]


def _parse_dataset(raw: dict[str, Any]) -> CkanDataset:
    resources = tuple(
        CkanResource(
            name=str(item.get("name") or item.get("description") or item.get("id", "")),
            url=str(item.get("url", "")),
            format=str(item.get("format", "")).upper(),
            size=item.get("size") if isinstance(item.get("size"), int) else None,
        )
        for item in raw.get("resources", [])
    )
    organization = raw.get("organization") or {}
    return CkanDataset(
        id=str(raw.get("name", "")),
        title=str(raw.get("title", "")),
        organization=str(organization.get("title")) if organization else None,
        resources=resources,
    )


def search_datasets(
    client: HttpClient,
    catalog_url: str,
    query: str,
    *,
    rows: int = 10,
) -> list[CkanDataset]:
    """Full-text search of a CKAN catalogue."""
    result = _api(client, catalog_url, "package_search", {"q": query, "rows": rows})
    datasets = [_parse_dataset(item) for item in result.get("results", [])]
    logger.info("CKAN search %r on %s: %d hit(s) of %s total",
                query, catalog_url, len(datasets), result.get("count"))
    return datasets


def get_dataset(client: HttpClient, catalog_url: str, dataset_id: str) -> CkanDataset:
    """Fetch one dataset by its CKAN name/id."""
    result = _api(client, catalog_url, "package_show", {"id": dataset_id})
    return _parse_dataset(result)


def resolve_resource_url(
    client: HttpClient,
    catalog_url: str,
    *,
    dataset_id: str | None = None,
    resource_url: str | None = None,
    format_filter: str | None = None,
) -> str:
    """Return an explicit download URL, preferring one the user already pinned."""
    if resource_url:
        return resource_url
    if not dataset_id:
        raise DataSourceError(
            "neither `resource_url` nor `dataset_id` is configured; run the "
            "downloader with --discover to search the catalogue first"
        )
    dataset = get_dataset(client, catalog_url, dataset_id)
    candidates = [
        resource
        for resource in dataset.resources
        if not format_filter or resource.format.upper() == format_filter.upper()
    ]
    if not candidates:
        raise DataSourceError(
            f"CKAN dataset {dataset_id} has no resource matching format "
            f"{format_filter!r}; available: "
            f"{sorted({r.format for r in dataset.resources})}"
        )
    if len(candidates) > 1:
        logger.warning(
            "CKAN dataset %s has %d matching resources; using the first. "
            "Pin `resource_url` in the config to be explicit.",
            dataset_id, len(candidates),
        )
    return candidates[0].url
