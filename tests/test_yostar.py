"""Tests for Japan YoStar launcher → resources.assets config resolution."""
import struct
from unittest.mock import MagicMock, patch

import pytest

from bagfd.crypto import create_key, encrypt_string, xor_inplace
from bagfd.yostar import (
    extract_game_main_from_resources,
    resolve_japan_server_info_url,
)


def _ok(json_data=None, content=None, text=None):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json.return_value = json_data
    if content is not None:
        resp.content = content
    if text is not None:
        resp.text = text
    return resp


def _build_resources_asset(plaintext_config: dict) -> bytes:
    """Build a resources.assets-like blob containing GameMainConfig."""
    import json

    key = create_key("GameMainConfig")
    utf16 = bytearray()
    text = json.dumps(plaintext_config, separators=(",", ":"))
    for ch in text:
        utf16.extend(struct.pack("<H", ord(ch)))
    xor_inplace(utf16, key)
    encrypted = bytes(utf16)

    marker = b"GameMainConfig\x00\x00"
    return b"PAD" + marker + struct.pack("<i", len(encrypted)) + encrypted + b"TAIL"


def _server_info_config(url: str) -> dict:
    server_key = create_key("ServerInfoDataUrl")
    return {
        encrypt_string("ServerInfoDataUrl", server_key): encrypt_string(url, server_key),
    }


class TestExtractGameMainFromResources:
    def test_extracts_and_decrypts_server_info_url(self):
        blob = _build_resources_asset(
            _server_info_config("https://example.yostar/server-info.json")
        )
        assert extract_game_main_from_resources(blob) == "https://example.yostar/server-info.json"

    def test_missing_pattern_raises(self):
        with pytest.raises(ValueError, match="GameMainConfig"):
            extract_game_main_from_resources(b"nope")


class TestResolveJapanServerInfoUrl:
    def test_launcher_path_skips_xapk(self):
        resources_blob = _build_resources_asset(
            _server_info_config("https://from-launcher/api")
        )
        # md5 of blob for hash check
        import hashlib
        asset_hash = hashlib.md5(resources_blob).hexdigest()

        base = _ok({
            "code": 200,
            "data": {
                "game_latest_version": "1.71.1",
                "game_latest_file_path": "BlueArchive/1.71.1",
            },
        })
        domain = _ok({"code": 200, "data": {"primary_cdn": "https://cdn.example/", "back_up_cdn": ""}})
        json_cfg = _ok({"code": 200, "data": {"url": "https://cdn.example/list.json"}})
        json_data = _ok({
            "source": "pkg/",
            "file": [
                {"path": "foo/bar", "hash": "aaa", "size": "1"},
                {"path": "x/resources.assets", "hash": asset_hash, "size": str(len(resources_blob))},
            ],
        })
        asset = _ok(content=resources_blob)

        def get_side(url, *args, **kwargs):
            if url.endswith("/api/launcher/game/config"):
                return base
            if "download/cdn" in url:
                return domain
            if "/api/launcher/game/config/json" in url:
                return json_cfg
            if url == "https://cdn.example/list.json":
                return json_data
            if url == "https://cdn.example/pkg/x/resources.assets":
                return asset
            raise AssertionError(f"unexpected GET {url}")

        session = MagicMock()
        session.get.side_effect = get_side

        with patch("bagfd.yostar._japan_api_url_from_xapk") as xapk:
            version, api_url = resolve_japan_server_info_url(session)
            xapk.assert_not_called()

        assert version == "1.71.1"
        assert api_url == "https://from-launcher/api"

    def test_falls_back_to_xapk_when_launcher_fails(self):
        session = MagicMock()
        session.get.side_effect = Exception("launcher down")

        with patch(
            "bagfd.yostar._japan_api_url_from_xapk",
            return_value=("1.70.0", "https://from-xapk/api"),
        ) as xapk:
            version, api_url = resolve_japan_server_info_url(session)
            xapk.assert_called_once()

        assert version == "1.70.0"
        assert api_url == "https://from-xapk/api"
