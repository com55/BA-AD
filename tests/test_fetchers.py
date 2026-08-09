"""
Unit tests for bagfd.fetchers — the maintenance-defer fallback.

These exercise the path where the game server is mid version-update: a new
version is detected but the catalog can't be fetched. The fetcher must keep the
cached catalog, park a defer window, and return "no new version" instead of
raising.
"""
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import requests

from bagfd.database import (
    get_cached_japan_api_url,
    get_game_files,
    get_stored_version,
    init_database,
    save_game_files,
    set_cached_japan_api_url,
    update_version,
)
from bagfd.fetchers import fetch_global_android, fetch_japan_servers


def _read_defer(db, platform):
    import sqlite3
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT defer_until FROM versions WHERE platform = ?", (platform,)).fetchone()
    conn.close()
    return row[0] if row else None


def _read_last_check(db, platform):
    import sqlite3
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT last_check FROM versions WHERE platform = ?", (platform,)).fetchone()
    conn.close()
    return row[0] if row else None


def _set_last_check(db, platform, when: datetime):
    import sqlite3
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE versions SET last_check = ? WHERE platform = ?",
        (when.isoformat(), platform),
    )
    conn.commit()
    conn.close()


def _ok(text=None, content=None, json_exc=None):
    """A fake response whose raise_for_status passes."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if text is not None:
        resp.text = text
    if content is not None:
        resp.content = content
    if json_exc is not None:
        resp.json.side_effect = json_exc
    return resp


class TestGlobalDefer:
    def test_catalog_failure_defers_and_keeps_stale(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "global-android", "1.0.0")  # stale catalog to fall back to

        version_resp = _ok(text="Blue Archive 1.2.3")
        # The catalog POST returns an empty body -> .json() blows up.
        bad_resp = _ok(json_exc=json.JSONDecodeError("Expecting value", "", 0))

        session = MagicMock()
        session.get.return_value = version_resp
        session.post.return_value = bad_resp

        # force=True to bypass the check interval and enter the catalog fetch.
        result = fetch_global_android(session, db, force=True)

        assert result is False                              # reported as no-new-version, no raise
        assert get_stored_version(db, "global-android") == "1.0.0"  # stale kept, not bumped to 1.2.3
        assert _read_defer(db, "global-android") is not None        # defer window parked


class TestJapanDefer:
    def test_catalog_failure_defers_both_platforms(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "japan-android", "1.69.0")
        update_version(db, "japan-windows", "1.69.0")

        bad_api = _ok(json_exc=json.JSONDecodeError("Expecting value", "", 0))
        session = MagicMock()
        session.get.return_value = bad_api

        with patch(
            "bagfd.yostar.resolve_japan_server_info_url",
            return_value=("1.70.436321", "https://fake/api"),
        ):
            results = fetch_japan_servers(session, db, force=True)

        assert results == {"japan-android": False, "japan-windows": False}
        assert get_stored_version(db, "japan-android") == "1.69.0"
        assert _read_defer(db, "japan-android") is not None
        assert _read_defer(db, "japan-windows") is not None


class TestJapanApiUrlCacheAndHotfix:
    """Cached API URL skips resources.assets/XAPK, but catalog lookup still runs."""

    def _mock_session(self, version_resp, addressable_resp, android_bundle_resp, windows_bundle_resp, api_url):
        requested = []

        def get_side(url, *args, **kwargs):
            requested.append(url)
            if (
                "api-launcher-jp.yo-star.com/api/launcher/game/config" in url
                and "json" not in url
                and "cdn" not in url
            ):
                return version_resp
            if url == api_url:
                return addressable_resp
            if "Android_PatchPack" in url:
                return android_bundle_resp
            if "Windows_PatchPack" in url:
                return windows_bundle_resp
            raise AssertionError(f"unexpected GET {url}")

        session = MagicMock()
        session.get.side_effect = get_side
        return session, requested

    def test_same_version_reuses_api_url_but_still_detects_hotfix(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "japan-android", "1.70.436321", is_new_version=True)
        update_version(db, "japan-windows", "1.70.436321", is_new_version=True)
        set_cached_japan_api_url(db, "1.70.436321", "https://fake/api")

        save_game_files(db, "japan_android", [
            ("Android_Pack_0.zip", "http://cdn/Android_PatchPack/Android_Pack_0.zip", "crc32", "111", 1000, "[]"),
        ])
        save_game_files(db, "japan_windows", [
            ("Windows_Pack_0.zip", "http://cdn/Windows_PatchPack/Windows_Pack_0.zip", "crc32", "222", 1000, "[]"),
        ])

        version_resp = _ok()
        version_resp.json.return_value = {
            "code": 200,
            "data": {"game_latest_version": "1.70.436321", "game_latest_file_path": "x"},
        }
        addressable = _ok()
        addressable.json.return_value = {
            "ConnectionGroups": [{"OverrideConnectionGroups": [{}, {"AddressablesCatalogUrlRoot": "http://cdn"}]}]
        }
        android_bundle = _ok()
        android_bundle.json.return_value = {"FullPatchPacks": [
            {"PackName": "Android_Pack_0.zip", "Crc": 999, "PackSize": 1000, "BundleFiles": []}
        ], "UpdatePacks": []}
        windows_bundle = _ok()
        windows_bundle.json.return_value = {"FullPatchPacks": [
            {"PackName": "Windows_Pack_0.zip", "Crc": 222, "PackSize": 1000, "BundleFiles": []}
        ], "UpdatePacks": []}

        session, requested = self._mock_session(
            version_resp, addressable, android_bundle, windows_bundle, "https://fake/api"
        )
        results = fetch_japan_servers(session, db, force=False, check_interval=timedelta(seconds=-1))

        assert results == {"japan-android": True, "japan-windows": False}
        assert not any("resources.assets" in u or u.endswith(".xapk") for u in requested)
        rows = {path: hv for path, _u, _ht, hv, _s, _bf in get_game_files(db, "japan_android")}
        assert rows["Android_Pack_0.zip"] == "999"

    def test_identical_catalog_returns_false(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "japan-android", "1.70.436321", is_new_version=True)
        update_version(db, "japan-windows", "1.70.436321", is_new_version=True)
        set_cached_japan_api_url(db, "1.70.436321", "https://fake/api")

        bundle_files_json = json.dumps(["a.bundle", "b.bundle"])
        save_game_files(db, "japan_android", [
            ("Android_Pack_0.zip", "http://cdn/Android_PatchPack/Android_Pack_0.zip", "crc32", "111", 1000, bundle_files_json),
        ])
        save_game_files(db, "japan_windows", [
            ("Windows_Pack_0.zip", "http://cdn/Windows_PatchPack/Windows_Pack_0.zip", "crc32", "222", 2000, "[]"),
        ])

        version_resp = _ok()
        version_resp.json.return_value = {
            "code": 200,
            "data": {"game_latest_version": "1.70.436321", "game_latest_file_path": "x"},
        }
        addressable = _ok()
        addressable.json.return_value = {
            "ConnectionGroups": [{"OverrideConnectionGroups": [{}, {"AddressablesCatalogUrlRoot": "http://cdn"}]}]
        }
        android_bundle = _ok()
        android_bundle.json.return_value = {"FullPatchPacks": [
            {"PackName": "Android_Pack_0.zip", "Crc": 111, "PackSize": 1000,
             "BundleFiles": [{"Name": "a.bundle"}, {"Name": "b.bundle"}]}
        ], "UpdatePacks": []}
        windows_bundle = _ok()
        windows_bundle.json.return_value = {"FullPatchPacks": [
            {"PackName": "Windows_Pack_0.zip", "Crc": 222, "PackSize": 2000, "BundleFiles": []}
        ], "UpdatePacks": []}

        session, _requested = self._mock_session(
            version_resp, addressable, android_bundle, windows_bundle, "https://fake/api"
        )
        results = fetch_japan_servers(session, db, force=False, check_interval=timedelta(seconds=-1))
        assert results == {"japan-android": False, "japan-windows": False}


class TestJapanForceBypassesApiUrlCache:
    def test_force_resolves_fresh_server_info(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "japan-android", "1.70.436321", is_new_version=True)
        update_version(db, "japan-windows", "1.70.436321", is_new_version=True)
        set_cached_japan_api_url(db, "1.70.436321", "https://stale-cached/api")

        addressable = _ok()
        addressable.json.return_value = {
            "ConnectionGroups": [{"OverrideConnectionGroups": [{}, {"AddressablesCatalogUrlRoot": "http://cdn"}]}]
        }
        bundle = _ok()
        bundle.json.return_value = {"FullPatchPacks": [], "UpdatePacks": []}

        def get_side(url, *args, **kwargs):
            if url == "https://fresh/api":
                return addressable
            if "PatchPack" in url:
                return bundle
            raise AssertionError(f"unexpected GET {url}")

        session = MagicMock()
        session.get.side_effect = get_side

        with patch(
            "bagfd.yostar.resolve_japan_server_info_url",
            return_value=("1.70.436321", "https://fresh/api"),
        ):
            fetch_japan_servers(session, db, force=True)

        assert get_cached_japan_api_url(db) == ("1.70.436321", "https://fresh/api")


class TestJapanUpdateVersionOnlyDuePlatforms:
    """Non-due platforms must keep their last_check so the interval is not reset."""

    def test_skips_update_version_for_platforms_not_due(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "japan-android", "1.70.436321", is_new_version=True)
        update_version(db, "japan-windows", "1.70.436321", is_new_version=True)
        set_cached_japan_api_url(db, "1.70.436321", "https://fake/api")

        # Only android is due; windows was checked recently.
        _set_last_check(db, "japan-android", datetime.now() - timedelta(hours=5))
        recent = datetime.now() - timedelta(minutes=10)
        _set_last_check(db, "japan-windows", recent)
        windows_last_check_before = _read_last_check(db, "japan-windows")

        save_game_files(db, "japan_android", [
            ("Android_Pack_0.zip", "http://cdn/Android_PatchPack/Android_Pack_0.zip", "crc32", "111", 1000, "[]"),
        ])
        save_game_files(db, "japan_windows", [
            ("Windows_Pack_0.zip", "http://cdn/Windows_PatchPack/Windows_Pack_0.zip", "crc32", "222", 1000, "[]"),
        ])

        version_resp = _ok()
        version_resp.json.return_value = {
            "code": 200,
            "data": {"game_latest_version": "1.70.436321", "game_latest_file_path": "x"},
        }
        addressable = _ok()
        addressable.json.return_value = {
            "ConnectionGroups": [{"OverrideConnectionGroups": [{}, {"AddressablesCatalogUrlRoot": "http://cdn"}]}]
        }
        android_bundle = _ok()
        android_bundle.json.return_value = {"FullPatchPacks": [
            {"PackName": "Android_Pack_0.zip", "Crc": 111, "PackSize": 1000, "BundleFiles": []}
        ], "UpdatePacks": []}
        windows_bundle = _ok()
        windows_bundle.json.return_value = {"FullPatchPacks": [
            {"PackName": "Windows_Pack_0.zip", "Crc": 222, "PackSize": 1000, "BundleFiles": []}
        ], "UpdatePacks": []}

        session = MagicMock()

        def get_side(url, *args, **kwargs):
            if (
                "api-launcher-jp.yo-star.com/api/launcher/game/config" in url
                and "json" not in url
                and "cdn" not in url
            ):
                return version_resp
            if url == "https://fake/api":
                return addressable
            if "Android_PatchPack" in url:
                return android_bundle
            if "Windows_PatchPack" in url:
                return windows_bundle
            raise AssertionError(f"unexpected GET {url}")

        session.get.side_effect = get_side

        fetch_japan_servers(session, db, force=False, check_interval=timedelta(hours=4))

        assert _read_last_check(db, "japan-windows") == windows_last_check_before
        # Due platform should have been refreshed.
        assert (
            datetime.fromisoformat(_read_last_check(db, "japan-android"))
            > datetime.fromisoformat(windows_last_check_before)
        )
