"""YoStar JP launcher client — fetch GameMainConfig via resources.assets.

Mirrors Deathemonic/BA-AD v3: prefer the launcher CDN (~60MB resources.assets)
over downloading the full Japan XAPK (~200MB). Falls back to the legacy
PureAPK + XAPK path when the launcher flow fails.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import time
from io import BytesIO
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)

YOSTAR_BASE_URL = "https://api-launcher-jp.yo-star.com"
YOSTAR_GAME_BASE_CONFIG_PATH = "/api/launcher/game/config"
YOSTAR_GAME_JSON_CONFIG_PATH = "/api/launcher/game/config/json"
YOSTAR_DOMAIN_PATH = "/api/launcher/advanced/game/download/cdn"
YOSTAR_GAME_TAG = "BlueArchive_JP"
YOSTAR_SIGNATURE_DATA = "DE7108E9B2842FD460F4777702727869"
YOSTAR_VERSION = "1.7.2"

# Layout: b"GameMainConfig\0\0" + i32le size + encrypted payload
GAME_CONFIG_PATTERN = bytes([
    0x47, 0x61, 0x6D, 0x65, 0x4D, 0x61, 0x69, 0x6E, 0x43, 0x6F, 0x6E, 0x66, 0x69, 0x67,
    0x00, 0x00, 0x92, 0x03, 0x00, 0x00,
])

PUREAPK_JAPAN_URL = (
    "https://api.pureapk.com/m/v3/cms/app_version"
    "?hl=en-US&package_name=com.YostarJP.BlueArchive"
)

_VERSION_RE = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_XAPK_URL_RE = re.compile(
    r"(X?APKJ)..(https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}"
    r"\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))"
)


def _md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _yostar_authorization() -> str:
    head = {
        "game_tag": YOSTAR_GAME_TAG,
        "time": int(time.time() * 1000),
        "version": YOSTAR_VERSION,
    }
    head_json = json.dumps(head, separators=(",", ":"))
    sign = _md5_hex(head_json + YOSTAR_SIGNATURE_DATA)
    return json.dumps({"head": head, "sign": sign}, separators=(",", ":"))


def _yostar_request(session: requests.Session, endpoint: str) -> Any:
    url = f"{YOSTAR_BASE_URL}{endpoint}"
    response = session.get(url, headers={"Authorization": _yostar_authorization()})
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 200:
        raise ValueError(f"YoStar API error code={payload.get('code')} for {endpoint}")
    return payload["data"]


def get_yostar_base_config(session: requests.Session) -> dict:
    """Return launcher game base config (includes game_latest_version)."""
    data = _yostar_request(session, YOSTAR_GAME_BASE_CONFIG_PATH)
    if not isinstance(data, dict) or "game_latest_version" not in data:
        raise ValueError("YoStar base config missing game_latest_version")
    return data


def get_resources_asset_url(session: requests.Session, base_config: dict) -> tuple[str, dict]:
    """Resolve the CDN URL and file info for resources.assets."""
    domain = _yostar_request(session, YOSTAR_DOMAIN_PATH)
    primary_cdn = domain.get("primary_cdn")
    if not primary_cdn:
        raise ValueError("YoStar domain missing primary_cdn")

    version = base_config["game_latest_version"]
    file_path = base_config["game_latest_file_path"]
    encoded_path = quote(file_path, safe="")
    endpoint = (
        f"{YOSTAR_GAME_JSON_CONFIG_PATH}?version={quote(version, safe='')}"
        f"&file_path={encoded_path}"
    )
    json_config = _yostar_request(session, endpoint)
    list_url = json_config.get("url")
    if not list_url:
        raise ValueError("YoStar json config missing url")

    list_resp = session.get(list_url)
    list_resp.raise_for_status()
    json_data = list_resp.json()
    files = json_data.get("file") or []
    source = json_data.get("source") or ""

    resources = next(
        (
            f for f in files
            if isinstance(f, dict) and str(f.get("path", "")).endswith("/resources.assets")
        ),
        None,
    )
    if resources is None:
        raise ValueError("resources.assets not found in YoStar file list")

    download_url = f"{primary_cdn}{source}{resources['path']}"
    return download_url, resources


def _decrypt_game_main_payload(encrypted_data: bytes) -> str:
    """Decrypt GameMainConfig bytes -> ServerInfoDataUrl (same crypto as XAPK path)."""
    import base64

    from .crypto import create_key, decrypt_string, encrypt_string, extract_json_from_string

    encoded_data = base64.b64encode(encrypted_data).decode("ascii")
    game_config_key = create_key("GameMainConfig")
    server_data_key = create_key("ServerInfoDataUrl")
    decrypted_data = decrypt_string(encoded_data, game_config_key)

    try:
        loaded_data = extract_json_from_string(decrypted_data)
    except Exception:
        last_brace = decrypted_data.rfind("}")
        if last_brace > 0:
            loaded_data = json.loads(decrypted_data[: last_brace + 1])
        else:
            raise

    if not isinstance(loaded_data, dict):
        raise ValueError("Decrypted config is not a JSON object")

    encrypted_key = encrypt_string("ServerInfoDataUrl", server_data_key)
    encrypted_value = loaded_data.get(encrypted_key)
    if not encrypted_value:
        raise ValueError("Key 'ServerInfoDataUrl' not found in decrypted config")

    result = decrypt_string(encrypted_value, server_data_key)
    if not result or not result.strip():
        raise ValueError("Decrypted ServerInfoDataUrl is empty")
    return result


def extract_game_main_from_resources(blob: bytes) -> str:
    """Find GameMainConfig in a resources.assets blob and return ServerInfoDataUrl."""
    pos = blob.find(GAME_CONFIG_PATTERN)
    if pos >= 0:
        size = struct.unpack_from("<i", GAME_CONFIG_PATTERN, len(GAME_CONFIG_PATTERN) - 4)[0]
        data_start = pos + len(GAME_CONFIG_PATTERN)
    else:
        marker = b"GameMainConfig\x00\x00"
        pos = blob.find(marker)
        if pos < 0:
            raise ValueError("GameMainConfig pattern not found in resources.assets")
        size_offset = pos + len(marker)
        if size_offset + 4 > len(blob):
            raise ValueError("GameMainConfig payload size out of range")
        size = struct.unpack_from("<i", blob, size_offset)[0]
        data_start = size_offset + 4

    if size <= 0 or data_start + size > len(blob):
        raise ValueError("GameMainConfig payload size out of range")

    return _decrypt_game_main_payload(blob[data_start: data_start + size])


def _japan_api_url_from_launcher(session: requests.Session) -> tuple[str, str]:
    """Return (version, server_info_url) via YoStar launcher + resources.assets."""
    base = get_yostar_base_config(session)
    version = base["game_latest_version"]
    logger.info("YoStar JP version: %s", version)

    download_url, file_info = get_resources_asset_url(session, base)
    logger.info(
        "Downloading resources.assets (%s bytes expected)...",
        file_info.get("size", "?"),
    )
    resp = session.get(download_url, stream=True)
    resp.raise_for_status()
    blob = resp.content

    expected_hash = (file_info.get("hash") or "").strip().lower()
    # YoStar currently returns a decimal CRC64-like value, not an MD5 hex digest.
    # Only verify when the catalog gives a classic 32-char hex MD5.
    if len(expected_hash) == 32 and all(c in "0123456789abcdef" for c in expected_hash):
        actual = hashlib.md5(blob).hexdigest()
        if actual != expected_hash:
            raise ValueError(
                f"resources.assets hash mismatch: expected {expected_hash}, got {actual}"
            )
    elif expected_hash:
        logger.debug(
            "Skipping resources.assets hash verify (non-MD5 hash=%s)", expected_hash
        )

    api_url = extract_game_main_from_resources(blob)
    return version, api_url


def _japan_api_url_from_xapk(session: requests.Session) -> tuple[str, str]:
    """Legacy fallback: PureAPK version + XAPK GameMainConfig extract."""
    from .fetchers import _extract_japan_api_url

    logger.warning("Falling back to PureAPK + XAPK for Japan server info URL")
    response = session.get(PUREAPK_JAPAN_URL)
    response.raise_for_status()

    match = _VERSION_RE.search(response.text)
    if not match:
        raise ValueError("Could not extract version from PureAPK Japan")
    version = match.group(0)

    url_match = _XAPK_URL_RE.search(response.text)
    if not url_match or len(url_match.groups()) < 2:
        raise ValueError("Could not extract APK URL from PureAPK Japan")

    download_url = url_match.group(2)
    logger.info("Downloading Japan XAPK...")
    xapk_data = BytesIO(session.get(download_url, stream=True).content)
    api_url = _extract_japan_api_url(session, xapk_data)
    return version, api_url


def resolve_japan_server_info_url(session: requests.Session) -> tuple[str, str]:
    """Resolve (version, ServerInfoDataUrl), preferring launcher over XAPK."""
    try:
        return _japan_api_url_from_launcher(session)
    except Exception as exc:
        logger.warning("YoStar launcher path failed (%s); trying XAPK fallback", exc)
        return _japan_api_url_from_xapk(session)
