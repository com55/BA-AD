"""
Database operations for Blue Archive game files.
"""

import sqlite3
import logging
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional, Dict, Tuple

from .downloader import _hash_matches

if TYPE_CHECKING:
    from .downloader import PathLockManager

logger = logging.getLogger(__name__)

VALID_TABLE_NAMES = {"global_android", "japan_android", "japan_windows"}


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection tuned for frequent, concurrent-ish access.

    Enables WAL journaling (better read/write concurrency, fewer "database is
    locked" errors) and a busy timeout so writers wait briefly instead of
    failing immediately. WAL is persisted in the database file, so setting it
    on every connection is a cheap no-op after the first time.
    """
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _validate_table_name(table_name: str) -> None:
    """Validate table name to prevent SQL injection.
    
    Args:
        table_name: Table name to validate.
        
    Raises:
        ValueError: If table name is not in the whitelist.
    """
    if table_name not in VALID_TABLE_NAMES:
        raise ValueError(f"Invalid table name: {table_name}")


def init_database(db_path: Path) -> None:
    """Initialize database and tables.
    
    Creates the versions table for storing platform version information,
    and platform-specific tables for storing game files metadata.
    
    Args:
        db_path: Path to the SQLite database file.
    """
    # auto_vacuum is baked into the database header at creation, so it must be
    # set before anything else writes the header (incl. switching to WAL) and
    # before any table exists. Use a raw connection here and set it first.
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA auto_vacuum=FULL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA journal_mode=WAL")
    cursor = conn.cursor()

    # Table: versions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS versions (
            platform TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            last_check TIMESTAMP NOT NULL,
            last_update TIMESTAMP NOT NULL,
            defer_until TIMESTAMP
        )
    """)

    # Migration: add defer_until to databases created before it existed.
    # CREATE TABLE IF NOT EXISTS won't add a column to a pre-existing table,
    # so ALTER it in. Idempotent — the OperationalError fires once the column
    # is already present (fresh DBs created by the statement above).
    try:
        cursor.execute("ALTER TABLE versions ADD COLUMN defer_until TIMESTAMP")
    except sqlite3.OperationalError:
        pass

    # Create separate tables for each platform. `path` is the primary key — no
    # surrogate id / AUTOINCREMENT, so nothing climbs across catalog refreshes.
    for table_name in ["global_android", "japan_android", "japan_windows"]:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                path TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                hash_type TEXT NOT NULL,
                hash_value TEXT NOT NULL,
                size INTEGER NOT NULL,
                bundle_files TEXT
            )
        """)

    # Caches the API URL decrypted from the Japan XAPK, keyed to the app
    # version it was extracted from — lets a same-version recheck skip
    # re-downloading the (large) XAPK while still hitting the live API each
    # time to pick up hotfix bundles pushed without a version bump.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS japan_api_cache (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version TEXT NOT NULL,
            api_url TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def get_table_name(platform: str) -> str:
    """Get table name for platform.
    
    Args:
        platform: Platform identifier ('global_android', 'japan_android', or 'japan_windows').
        
    Returns:
        Database table name for the platform.
    """
    table_map = {
        "global-android": "global_android",
        "japan-android": "japan_android",
        "japan-windows": "japan_windows",
    }
    return table_map.get(platform, "")


def should_check_version(db_path: Path, platform: str, force: bool = False, 
                        check_interval: timedelta = timedelta(hours=4)) -> bool:
    """Check if version check is needed.
    
    Determines whether to fetch a new version based on the last check time
    and the specified check interval.
    
    Args:
        db_path: Path to the SQLite database.
        platform: Platform identifier.
        force: Force version check regardless of interval.
        check_interval: Minimum time between checks.
        
    Returns:
        True if version should be checked, False otherwise.
    """
    if force:
        return True
    
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT last_check, defer_until FROM versions WHERE platform = ?", (platform,))
    result = cursor.fetchone()
    conn.close()

    if not result:
        return True

    last_check, defer_until = result
    # A failed catalog fetch parks a short defer window here so we keep serving
    # the cached catalog instead of hammering a server that's down for
    # maintenance. Honour it before the normal interval check.
    if defer_until and datetime.now() < datetime.fromisoformat(defer_until):
        return False

    return datetime.now() - datetime.fromisoformat(last_check) > check_interval


def get_stored_version(db_path: Path, platform: str) -> Optional[str]:
    """Get stored version from database.
    
    Args:
        db_path: Path to the SQLite database.
        platform: Platform identifier.
        
    Returns:
        Stored version string, or None if not found.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT version FROM versions WHERE platform = ?", (platform,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def update_version(db_path: Path, platform: str, version: str, is_new_version: bool = False) -> None:
    """Update version in database.
    
    Args:
        db_path: Path to the SQLite database.
        platform: Platform identifier.
        version: New version string.
        is_new_version: Whether this is a new version (updates last_update timestamp).
    """
    now = datetime.now().isoformat()

    conn = _connect(db_path)
    cursor = conn.cursor()

    # INSERT OR REPLACE rewrites the whole row, so an unspecified defer_until
    # resets to NULL — a successful version update naturally clears any pending
    # defer window. Keep defer_until out of the column list to preserve that.
    if is_new_version:
        cursor.execute("""
            INSERT OR REPLACE INTO versions (platform, version, last_check, last_update)
            VALUES (?, ?, ?, ?)
        """, (platform, version, now, now))
    else:
        cursor.execute("""
            INSERT OR REPLACE INTO versions (platform, version, last_check, last_update)
            VALUES (?, ?, ?, COALESCE((SELECT last_update FROM versions WHERE platform = ?), ?))
        """, (platform, version, now, platform, now))
    
    conn.commit()
    conn.close()


def get_cached_japan_api_url(db_path: Path) -> Optional[Tuple[str, str]]:
    """Get the cached Japan API URL and the version it was extracted for.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        ``(version, api_url)``, or None if nothing is cached yet.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT version, api_url FROM japan_api_cache WHERE id = 1")
    result = cursor.fetchone()
    conn.close()
    return (result[0], result[1]) if result else None


def set_cached_japan_api_url(db_path: Path, version: str, api_url: str) -> None:
    """Cache the Japan API URL extracted from the XAPK for ``version``.

    Args:
        db_path: Path to the SQLite database.
        version: App version the API URL was extracted from.
        api_url: Decrypted API URL.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO japan_api_cache (id, version, api_url)
        VALUES (1, ?, ?)
    """, (version, api_url))
    conn.commit()
    conn.close()


def set_defer(db_path: Path, platform: str, until: datetime) -> None:
    """Defer ``platform``'s next version check until ``until``.

    Stamps ``defer_until`` on the existing version row so `should_check_version`
    keeps serving the cached catalog through a transient outage instead of
    re-fetching. No-op if the platform has no row yet — a platform that has
    never been fetched has no stale catalog to fall back on.

    Args:
        db_path: Path to the SQLite database.
        platform: Platform identifier.
        until: Wall-clock time to keep deferring until.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE versions SET defer_until = ? WHERE platform = ?",
        (until.isoformat(), platform),
    )
    conn.commit()
    conn.close()


def clear_defer(db_path: Path, platform: str) -> None:
    """Clear any active defer window for ``platform``.

    Args:
        db_path: Path to the SQLite database.
        platform: Platform identifier.
    """
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE versions SET defer_until = NULL WHERE platform = ?",
        (platform,),
    )
    conn.commit()
    conn.close()


def clear_cache_for_platform(
    cache_dir: Path, platform: str, locks: "PathLockManager | None" = None
) -> None:
    """Clear a platform's cached files, without touching ``cache_dir`` itself.

    Removes everything inside ``cache_dir/<platform>/`` only. ``cache_dir`` is
    left alone since it may hold unrelated files or other platforms' folders.

    Args:
        cache_dir: Base cache directory path.
        platform: Platform identifier.
        locks: Optional `PathLockManager`. When given, each cached file is
            removed only if its per-file lock can be taken without blocking, so
            a file another client/thread is currently downloading or extracting
            is skipped (and reclaimed on a later pass) instead of being deleted
            mid-use — which on Windows raises ``WinError 32``.
    """
    platform_cache = cache_dir / platform
    if not platform_cache.exists():
        return
    for entry in platform_cache.iterdir():
        _remove_cache_entry(entry, locks)
    logger.info(f"Cache cleared: {platform}")


def prune_stale_cache(
    cache_dir: Path,
    platform: str,
    valid_hashes: Dict[str, Tuple[str, str]],
    locks: "PathLockManager | None" = None,
) -> None:
    """Remove cached files under ``cache_dir/<platform>/`` no longer valid.

    Unlike `clear_cache_for_platform` (which wipes everything), this keeps any
    cached file whose name is still in ``valid_hashes`` *and* whose on-disk
    hash still matches — so a catalog refresh only evicts the files that
    actually changed or dropped out, not the whole cache.

    Args:
        cache_dir: Base cache directory path.
        platform: Platform identifier.
        valid_hashes: Map of cache filename -> (hash_type, hash_value) built
            from the current catalog rows.
        locks: Optional `PathLockManager`, passed through to skip files
            another client/thread currently holds (see `clear_cache_for_platform`).
    """
    platform_cache = cache_dir / platform
    if not platform_cache.exists():
        return
    for entry in platform_cache.iterdir():
        if not entry.is_file():
            continue
        _prune_cache_entry(entry, valid_hashes.get(entry.name), locks)


def _prune_cache_entry(
    entry: Path, valid: Optional[Tuple[str, str]], locks: "PathLockManager | None"
) -> None:
    """Delete ``entry`` unless it's still listed in the catalog with a matching hash.

    Hashing happens inside the same lock used for deletion, so a file a
    downloader is actively writing is skipped rather than hashed mid-write.
    """
    if locks is not None:
        with locks.try_lock(entry) as acquired:
            if not acquired:
                logger.debug(f"Skipped in-use cache file: {entry.name}")
                return
            if valid is None or not _hash_matches(entry, valid[0], valid[1]):
                _delete_cache_entry(entry)
    else:
        if valid is None or not _hash_matches(entry, valid[0], valid[1]):
            _delete_cache_entry(entry)


def _remove_cache_entry(entry: Path, locks: "PathLockManager | None") -> None:
    """Delete one cache entry, guarded by ``locks`` and tolerant of in-use files."""
    if locks is not None and entry.is_file():
        with locks.try_lock(entry) as acquired:
            if not acquired:
                logger.debug(f"Skipped in-use cache file: {entry.name}")
                return
            _delete_cache_entry(entry)
    else:
        _delete_cache_entry(entry)


def _delete_cache_entry(entry: Path) -> None:
    """Remove a file or directory, logging instead of raising if it's in use."""
    try:
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    except OSError as e:
        logger.warning(f"Could not remove cache entry {entry.name}: {e}")


def save_game_files(db_path: Path, table_name: str, files: list) -> None:
    """Save game files to database.
    
    Replaces all existing files in the table with the provided list.
    
    Args:
        db_path: Path to the SQLite database.
        table_name: Target table name.
        files: List of file entries (tuples of (path, url, hash_type, hash_value, size, bundle_files)).
        
    Raises:
        ValueError: If table name is invalid.
    """
    _validate_table_name(table_name)
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name}")
    
    for path, url, hash_type, hash_value, size, bundle_files in files:
        cursor.execute(f"""
            INSERT INTO {table_name} (path, url, hash_type, hash_value, size, bundle_files)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (path, url, hash_type, hash_value, size, bundle_files))
    
    conn.commit()
    conn.close()


def clear_platform_db(db_path: Path, platform: str) -> None:
    """Clear all DB entries for a platform (table rows + version record).
    
    Raises:
        ValueError: If table name lookup fails.
    """
    table_name = get_table_name(platform)
    if not table_name:
        raise ValueError(f"Unknown platform: {platform}")
    _validate_table_name(table_name)
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM {table_name}")
    cursor.execute("DELETE FROM versions WHERE platform = ?", (platform,))
    conn.commit()
    conn.close()
    logger.info(f"DB cleared: {platform}")


def get_game_files(db_path: Path, table_name: str) -> list:
    """Get all game files from database.
    
    Args:
        db_path: Path to the SQLite database.
        table_name: Source table name.
        
    Returns:
        List of file entries (tuples of (path, url, hash_type, hash_value, size, bundle_files)).
        
    Raises:
        ValueError: If table name is invalid.
    """
    _validate_table_name(table_name)
    conn = _connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT path, url, hash_type, hash_value, size, bundle_files FROM {table_name}")
    results = cursor.fetchall()
    conn.close()
    return results
