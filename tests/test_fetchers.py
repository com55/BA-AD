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

        # PureAPK page text must satisfy both the version regex and the APK-url regex.
        pureapk = _ok(text="XAPKJ: https://example.com/app.xapk build 1.70.436321")
        xapk = _ok(content=b"fake-xapk-bytes")
        bad_api = _ok(json_exc=json.JSONDecodeError("Expecting value", "", 0))

        def get_side(url, *args, **kwargs):
            if "pureapk" in url:
                return pureapk
            if url.endswith(".xapk"):
                return xapk
            return bad_api  # the addressable api_url

        session = MagicMock()
        session.get.side_effect = get_side

        with patch("bagfd.fetchers._extract_japan_api_url", return_value="https://fake/api"):
            results = fetch_japan_servers(session, db, force=True)

        assert results == {"japan-android": False, "japan-windows": False}  # no new version, no raise
        assert get_stored_version(db, "japan-android") == "1.69.0"          # stale kept
        assert _read_defer(db, "japan-android") is not None
        assert _read_defer(db, "japan-windows") is not None


class TestGlobalSameVersionHotfix:
    """A due periodic check (not a version bump) must still pick up a
    same-version hotfix: the catalog rewrite must not be gated on
    `is_new_version`."""

    def test_same_version_content_change_is_detected(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "global-android", "1.90.439170", is_new_version=True)
        save_game_files(db, "global_android", [
            ("0/Android/aa/unchanged.bundle", "http://cdn/patch/0/Android/aa/unchanged.bundle", "md5", "aaa", 100, None),
            ("0/Android/aa/changed.bundle", "http://cdn/patch/0/Android/aa/changed.bundle", "md5", "old-hash", 200, None),
        ])

        version_resp = _ok(text="Blue Archive 1.90.439170")
        post_resp = _ok()
        post_resp.json.return_value = {"patch": {"resource_path": "http://cdn/patch/resource-data.json"}}
        resources_resp = _ok()
        resources_resp.json.return_value = {"resources": [
            {"resource_path": "0/Android/aa/unchanged.bundle", "resource_hash": "aaa", "resource_size": 100},
            {"resource_path": "0/Android/aa/changed.bundle", "resource_hash": "new-hash", "resource_size": 200},
            {"resource_path": "0/Android/aa/newfile.bundle", "resource_hash": "bbb", "resource_size": 50},
        ]}

        def get_side(url, *args, **kwargs):
            return version_resp if "pureapk" in url else resources_resp

        session = MagicMock()
        session.get.side_effect = get_side
        session.post.return_value = post_resp

        # force=False, but the interval is already elapsed -> a normal due
        # recheck, not a forced full re-fetch.
        result = fetch_global_android(session, db, force=False, check_interval=timedelta(seconds=-1))

        assert result is True
        rows = {path: hash_value for path, _url, _ht, hash_value, _size, _bf in get_game_files(db, "global_android")}
        assert rows["0/Android/aa/changed.bundle"] == "new-hash"
        assert rows["0/Android/aa/newfile.bundle"] == "bbb"
        assert get_stored_version(db, "global-android") == "1.90.439170"  # version itself unchanged

    def test_identical_catalog_returns_false(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "global-android", "1.90.439170", is_new_version=True)
        save_game_files(db, "global_android", [
            ("0/Android/aa/a.bundle", "http://cdn/patch/0/Android/aa/a.bundle", "md5", "aaa", 100, None),
        ])

        version_resp = _ok(text="Blue Archive 1.90.439170")
        post_resp = _ok()
        post_resp.json.return_value = {"patch": {"resource_path": "http://cdn/patch/resource-data.json"}}
        resources_resp = _ok()
        resources_resp.json.return_value = {"resources": [
            {"resource_path": "0/Android/aa/a.bundle", "resource_hash": "aaa", "resource_size": 100},
        ]}

        def get_side(url, *args, **kwargs):
            return version_resp if "pureapk" in url else resources_resp

        session = MagicMock()
        session.get.side_effect = get_side
        session.post.return_value = post_resp

        result = fetch_global_android(session, db, force=False, check_interval=timedelta(seconds=-1))

        assert result is False


class TestJapanApiUrlCacheAndHotfix:
    """Covers both halves of the Japan fix at once: the cached API URL must
    skip the XAPK download, but the live catalog lookup through it must still
    run every due check so a same-version hotfix is never missed."""

    def _mock_session(self, pureapk_resp, addressable_resp, android_bundle_resp, windows_bundle_resp, api_url):
        requested = []

        def get_side(url, *args, **kwargs):
            requested.append(url)
            if "pureapk" in url:
                return pureapk_resp
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

        pureapk = _ok(text="Blue Archive JP 1.70.436321")
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

        session, requested = self._mock_session(pureapk, addressable, android_bundle, windows_bundle, "https://fake/api")

        results = fetch_japan_servers(session, db, force=False, check_interval=timedelta(seconds=-1))

        assert results == {"japan-android": True, "japan-windows": False}
        assert not any(u.endswith(".xapk") for u in requested)  # api_url cache saved the XAPK download
        rows = {path: hash_value for path, _url, _ht, hash_value, _size, _bf in get_game_files(db, "japan_android")}
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

        pureapk = _ok(text="Blue Archive JP 1.70.436321")
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

        session, _requested = self._mock_session(pureapk, addressable, android_bundle, windows_bundle, "https://fake/api")

        results = fetch_japan_servers(session, db, force=False, check_interval=timedelta(seconds=-1))

        assert results == {"japan-android": False, "japan-windows": False}


class TestJapanForceBypassesApiUrlCache:
    def test_force_redownloads_apk_even_with_valid_cache(self, tmp_path):
        db = tmp_path / "catalog.db"
        init_database(db)
        update_version(db, "japan-android", "1.70.436321", is_new_version=True)
        update_version(db, "japan-windows", "1.70.436321", is_new_version=True)
        set_cached_japan_api_url(db, "1.70.436321", "https://stale-cached/api")

        pureapk = _ok(text="XAPKJ: https://example.com/app.xapk build 1.70.436321")
        xapk = _ok(content=b"fake-xapk-bytes")
        addressable = _ok()
        addressable.json.return_value = {
            "ConnectionGroups": [{"OverrideConnectionGroups": [{}, {"AddressablesCatalogUrlRoot": "http://cdn"}]}]
        }
        bundle = _ok()
        bundle.json.return_value = {"FullPatchPacks": [], "UpdatePacks": []}

        def get_side(url, *args, **kwargs):
            if "pureapk" in url:
                return pureapk
            if url.endswith(".xapk"):
                return xapk
            if url == "https://fresh/api":
                return addressable
            if "PatchPack" in url:
                return bundle
            # Would only be hit if the stale cached URL were used instead of
            # re-deriving a fresh one -- fail loudly rather than silently pass.
            raise AssertionError(f"unexpected GET {url}")

        session = MagicMock()
        session.get.side_effect = get_side

        with patch("bagfd.fetchers._extract_japan_api_url", return_value="https://fresh/api"):
            fetch_japan_servers(session, db, force=True)

        assert get_cached_japan_api_url(db) == ("1.70.436321", "https://fresh/api")
