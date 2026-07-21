"""
Platform-specific fetching logic for Blue Archive game files.
"""

import logging
import json
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple
from io import BytesIO
import requests

from .crypto import extract_json_from_string, create_key, encrypt_string, decrypt_string
from .database import (
    should_check_version, get_stored_version, update_version,
    get_table_name, save_game_files, set_defer, get_game_files,
    get_cached_japan_api_url, set_cached_japan_api_url,
)

logger = logging.getLogger(__name__)


def _row_key(row: tuple) -> tuple:
    """Normalize a catalog row for equality comparison (case-insensitive hash).

    `_hash_matches` (used elsewhere for cache verification) already treats
    hash strings as case-insensitive; the catalog-diff comparison here does
    the same, so a hash-casing-only change from the upstream API isn't
    mistaken for real content drift.
    """
    path, url, hash_type, hash_value, size, bundle_files = row
    return (path, url, hash_type, str(hash_value).lower(), size, bundle_files)


def _decrypt_japan_config(encrypted_data: bytes) -> str:
    """Decrypt Japan game config.
    
    Decrypts the encrypted game configuration data from Japan server,
    extracting the server API URL.
    
    Args:
        encrypted_data: Raw encrypted configuration bytes.
        
    Returns:
        Decrypted server API URL.
        
    Raises:
        ValueError: If decryption fails or required key not found.
    """
    import base64
    
    encoded_data = base64.b64encode(encrypted_data).decode('ascii')
    game_config_key = create_key("GameMainConfig")
    server_data_key = create_key("ServerInfoDataUrl")
    decrypted_data = decrypt_string(encoded_data, game_config_key)
    
    try:
        loaded_data = extract_json_from_string(decrypted_data)
    except Exception:
        last_brace = decrypted_data.rfind('}')
        if last_brace > 0:
            json_str = decrypted_data[:last_brace + 1]
            loaded_data = json.loads(json_str)
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


def _extract_japan_api_url(session: requests.Session, xapk_data: BytesIO) -> str:
    """Extract API URL from Japan XAPK.
    
    Extracts the game configuration from the XAPK file and decrypts it
    to get the server API URL.
    
    Args:
        session: Requests session for HTTP operations.
        xapk_data: BytesIO containing XAPK file data.
        
    Returns:
        Decrypted server API URL.
        
    Raises:
        ValueError: If game config not found in APK.
    """
    config_pattern = bytes([0x47, 0x61, 0x6D, 0x65, 0x4D, 0x61, 0x69, 0x6E, 0x43, 0x6F, 0x6E, 0x66, 0x69, 0x67,
                           0x00, 0x00, 0x92, 0x03, 0x00, 0x00])
    
    with zipfile.ZipFile(xapk_data, 'r') as xapk:
        unity_apk_data = xapk.read('UnityDataAssetPack.apk')
        
        with zipfile.ZipFile(BytesIO(unity_apk_data), 'r') as unity_apk:
            for filename in unity_apk.namelist():
                if 'assets/bin/Data/' in filename and not filename.endswith('/'):
                    data = unity_apk.read(filename)
                    offset = data.find(config_pattern)
                    
                    if offset != -1:
                        data_start = offset + len(config_pattern)
                        encrypted_data = data[data_start:data_start + 1024]
                        return _decrypt_japan_config(encrypted_data)
    
    raise ValueError("Could not find game config")


def fetch_global_android(session: requests.Session, db_path: Path,
                        force: bool = False, check_interval=None,
                        defer_on_failure: timedelta = timedelta(minutes=10)) -> bool:
    """Fetch the Global Android game-file catalog into the database.

    Re-fetches and rewrites the catalog rows every time a check is due (per
    ``check_interval``/``force``) — not only when the app version changes — so
    a hotfix bundle pushed under the same version is still picked up. Does not
    touch any download cache — cache invalidation is the caller's responsibility.

    If the catalog can't be fetched (e.g. the server is mid version-update and
    returns an empty body or an error), the cached catalog is kept untouched and
    the next check is deferred by ``defer_on_failure`` rather than raising — so a
    transient maintenance window degrades to serving the previous version.

    Args:
        session: Requests session with proper headers configured.
        db_path: Path to the SQLite database.
        force: Force fetch regardless of version check interval.
        check_interval: Version check interval (uses default if None).
        defer_on_failure: How long to keep serving the cached catalog before
            re-attempting, when a fetch fails (default 10 minutes).

    Returns:
        True if the catalog changed (new version or a same-version content
        change), False otherwise.
    """
    if check_interval is None:
        check_interval = timedelta(hours=4)
    
    GLOBAL_API_URL = "https://api-pub.nexon.com/patch/v1.1/version-check"
    PUREAPK_GLOBAL_URL = "https://api.pureapk.com/m/v3/cms/app_version?hl=en-US&package_name=com.nexon.bluearchive"
    
    platform = "global-android"
    
    if not should_check_version(db_path, platform, force, check_interval):
        logger.debug(f"Skipped {platform} (checked recently)")
        return False
    
    logger.info(f"Fetching {platform}...")
    
    # Get version from PureAPK
    response = session.get(PUREAPK_GLOBAL_URL)
    response.raise_for_status()
    
    version_pattern = re.compile(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)')
    match = version_pattern.search(response.text)
    
    if not match:
        raise ValueError("Could not extract version from PureAPK")
    
    version = match.group(0)
    stored_version = get_stored_version(db_path, platform)
    is_new_version = version != stored_version
    
    if is_new_version:
        logger.info(f"New version: {stored_version} → {version}")
    else:
        logger.info(f"Version unchanged: {version}")
    
    # Always re-fetch the resource list when due, regardless of whether the
    # version string changed — a hotfix bundle can be pushed under the same
    # version, and the pointer lookup below is the only place that freshness
    # signal can surface.
    try:
        build_number = version.split('.')[-1]
        payload = {
            "market_game_id": "com.nexon.bluearchive",
            "market_code": "playstore",
            "curr_build_version": version,
            "curr_build_number": build_number
        }

        addressable_resp = session.post(GLOBAL_API_URL, json=payload)
        addressable_resp.raise_for_status()
        addressable = addressable_resp.json()
        resource_path = addressable['patch']['resource_path']
        resources_resp = session.get(resource_path)
        resources_resp.raise_for_status()
        resources = resources_resp.json()

        catalog_url = resource_path.replace('/resource-data.json', '')

        table_name = get_table_name(platform)
        files_to_save = []

        for resource in resources.get('resources', []):
            if '/Android/' in resource['resource_path']:
                files_to_save.append((
                    resource['resource_path'],
                    f"{catalog_url}/{resource['resource_path']}",
                    'md5',
                    resource['resource_hash'],
                    resource['resource_size'],
                    None
                ))

        existing_files = get_game_files(db_path, table_name)
        catalog_changed = {_row_key(r) for r in existing_files} != {_row_key(r) for r in files_to_save}
        if catalog_changed:
            save_game_files(db_path, table_name, files_to_save)
            logger.info(f"Updated {platform}")
        else:
            logger.info(f"{platform} catalog unchanged")
    except (requests.RequestException, ValueError, KeyError) as exc:
        # The server is most likely mid version-update (empty body, 4xx, or
        # malformed JSON). Keep the cached catalog untouched, park a short
        # defer window, and report "no new version" so the caller serves the
        # previous version instead of crashing the whole conversion.
        until = datetime.now() + defer_on_failure
        logger.warning(
            "Global catalog fetch failed (%s); keeping cached catalog and "
            "deferring re-check until %s", exc, until.isoformat(timespec="seconds"),
        )
        set_defer(db_path, platform, until)
        return False

    update_version(db_path, platform, version, is_new_version)

    return is_new_version or catalog_changed


def fetch_japan_servers(session: requests.Session, db_path: Path,
                       force: bool = False, check_interval=None,
                       defer_on_failure: timedelta = timedelta(minutes=10)) -> Dict[str, bool]:
    """Fetch the Japan Android & Windows game-file catalogs into the database.

    Re-fetches and rewrites the catalog rows for both Japan platforms every
    time a check is due (per ``check_interval``/``force``) — not only when the
    app version changes — so a hotfix bundle pushed under the same version is
    still picked up. The XAPK is only re-downloaded and decrypted when the app
    version changes (or ``force``); at an unchanged version the cached API URL
    is reused, but the live catalog lookup through it still runs every time,
    since that's where a same-version hotfix would surface. Does not touch any
    download cache — cache invalidation is the caller's responsibility.

    If the catalog can't be fetched (e.g. the server is mid version-update and
    returns an empty body or an error), the cached catalog is kept untouched and
    the next check is deferred by ``defer_on_failure`` rather than raising — so a
    transient maintenance window degrades to serving the previous version
    instead of failing the whole conversion.

    Args:
        session: Requests session with proper headers configured.
        db_path: Path to the SQLite database.
        force: Force fetch regardless of version check interval. Also forces a
            fresh XAPK download/decrypt instead of reusing a cached API URL.
        check_interval: Version check interval (uses default if None).
        defer_on_failure: How long to keep serving the cached catalog before
            re-attempting, when a fetch fails (default 10 minutes).

    Returns:
        Dict mapping each Japan platform name to whether its catalog changed
        (new version or a same-version content change).
    """
    if check_interval is None:
        check_interval = timedelta(hours=4)

    PUREAPK_JAPAN_URL = "https://api.pureapk.com/m/v3/cms/app_version?hl=en-US&package_name=com.YostarJP.BlueArchive"

    results = {}
    due = {}
    catalog_changed = {}
    current_version = None

    japan_platforms = ["japan-android", "japan-windows"]

    # Check versions for both platforms
    for platform_name in japan_platforms:
        if not should_check_version(db_path, platform_name, force, check_interval):
            logger.debug(f"Skipped {platform_name} (checked recently)")
            results[platform_name] = False
            due[platform_name] = False
            continue

        logger.info(f"Checking {platform_name}...")
        due[platform_name] = True

        if not current_version:
            response = session.get(PUREAPK_JAPAN_URL)
            response.raise_for_status()

            version_pattern = re.compile(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)')
            match = version_pattern.search(response.text)

            if not match:
                raise ValueError("Could not extract version from PureAPK Japan")

            current_version = match.group(0)

        stored_version = get_stored_version(db_path, platform_name)
        is_new_version = current_version != stored_version

        if is_new_version:
            logger.info(f"New version: {stored_version} → {current_version}")
        else:
            logger.info(f"Version unchanged: {current_version}")

        results[platform_name] = is_new_version

    # Re-check the catalog whenever either platform is due, regardless of
    # whether the version changed.
    if any(due.values()) or force:
        try:
            cached = get_cached_japan_api_url(db_path)
            if not force and cached and cached[0] == current_version:
                api_url = cached[1]
                logger.info("Reusing cached API URL for version %s", current_version)
            else:
                logger.info("Downloading Japan APK...")

                response = session.get(PUREAPK_JAPAN_URL)
                url_pattern = re.compile(
                    r'(X?APKJ)..(https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))'
                )
                url_match = url_pattern.search(response.text)

                if not url_match or len(url_match.groups()) < 2:
                    raise ValueError("Could not extract APK URL")

                download_url = url_match.group(2)
                logger.info("Downloading XAPK...")

                xapk_data = BytesIO(session.get(download_url, stream=True).content)
                logger.info("Extracting config...")
                api_url = _extract_japan_api_url(session, xapk_data)
                set_cached_japan_api_url(db_path, current_version, api_url)

            logger.info("Fetching catalogs...")
            addressable_resp = session.get(api_url)
            addressable_resp.raise_for_status()
            addressable = addressable_resp.json()
            connection_groups = addressable.get("ConnectionGroups", [])
            if not connection_groups:
                raise ValueError("No ConnectionGroups in addressable response")

            override_groups = connection_groups[0].get("OverrideConnectionGroups", [])
            if len(override_groups) < 2:
                raise ValueError(f"Expected at least 2 OverrideConnectionGroups, got {len(override_groups)}")

            catalog_url = override_groups[1].get("AddressablesCatalogUrlRoot", "")
            if not catalog_url:
                raise ValueError("AddressablesCatalogUrlRoot not found or empty in OverrideConnectionGroups[1]")

            # Process both platforms
            for platform_name in japan_platforms:
                platform_key = "Windows" if platform_name == "japan-windows" else "Android"
                patch_pack = f"{platform_key}_PatchPack"
                logger.info(f"Downloading {platform_name} catalog...")

                bundle_url = f"{catalog_url}/{patch_pack}/BundlePackingInfo.json"
                bundle_resp = session.get(bundle_url)
                bundle_resp.raise_for_status()
                bundle_data = bundle_resp.json()

                table_name = get_table_name(platform_name)
                files_to_save = []

                all_packs = bundle_data.get('FullPatchPacks', []) + bundle_data.get('UpdatePacks', [])

                for pack in all_packs:
                    # Sorted so the JSON is stable regardless of the order the
                    # server lists members in — an unrelated reorder shouldn't
                    # register as a catalog change.
                    bundle_files = json.dumps(sorted(bf['Name'] for bf in pack.get('BundleFiles', [])))

                    files_to_save.append((
                        pack['PackName'],
                        f"{catalog_url}/{patch_pack}/{pack['PackName']}",
                        'crc32',
                        str(pack['Crc']),
                        pack['PackSize'],
                        bundle_files
                    ))

                existing_files = get_game_files(db_path, table_name)
                changed = {_row_key(r) for r in existing_files} != {_row_key(r) for r in files_to_save}
                catalog_changed[platform_name] = changed
                if changed:
                    save_game_files(db_path, table_name, files_to_save)
                    logger.info(f"Updated {platform_name}")
                else:
                    logger.info(f"{platform_name} catalog unchanged")
        except (requests.RequestException, ValueError, KeyError) as exc:
            # The server is most likely mid version-update (empty body, 4xx, or
            # malformed JSON). Keep the cached catalog untouched, park a short
            # defer window, and report "no new version" so the caller serves the
            # previous version instead of crashing the whole conversion.
            until = datetime.now() + defer_on_failure
            logger.warning(
                "Japan catalog fetch failed (%s); keeping cached catalog and "
                "deferring re-check until %s", exc, until.isoformat(timespec="seconds"),
            )
            for platform_name in japan_platforms:
                set_defer(db_path, platform_name, until)
                results[platform_name] = False
            return results

    # Update version info for all platforms
    if current_version:
        for platform_name in japan_platforms:
            update_version(db_path, platform_name, current_version, results[platform_name])

    return {
        platform_name: results.get(platform_name, False) or catalog_changed.get(platform_name, False)
        for platform_name in japan_platforms
    }
