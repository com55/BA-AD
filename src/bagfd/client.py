"""Blue Archive game-file client.

`BlueArchiveGameFilesDownloader` is the high-level entry point. It keeps a fixed
**data directory** (catalog DB + the JP zip cache) and exposes three
file-oriented operations, each scoped to a single platform:

- `query` — search the catalog, return rich `FileInfo` metadata.
- `get_latest_files` — make sure the latest matching files exist in a *cache
  directory* and return their paths there (download-or-reuse).
- `download` — download the latest matching files into a *user output
  directory* and return a `DownloadResult`.

Storage layout (all under ``data_dir``, except the per-call download dirs and
an optionally overridden catalog DB):

- ``data_dir/catalog.db``       — the file catalog (path overridable via ``db_path``,
                                  e.g. to keep it off shared/network-backed storage)
- ``data_dir/zip_cache/<plat>`` — cached JP zip packs (shared by download/get)
- ``cache_dir/<plat>``          — files returned by `get_latest_files`
                                  (default ``data_dir/download_cache``)
- ``output_dir``                — files delivered by `download` (default ``./download``)

Platform/verify/filter options are `StrEnum`s, so the enum members and their
string values ("global-android", "hash", "glob", …) are interchangeable.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import zipfile
from pathlib import Path

import requests

from .database import (
    clear_cache_for_platform,
    clear_platform_db,
    get_game_files,
    get_table_name,
    init_database,
    prune_stale_cache,
)
from .downloader import DownloadItem, PathLockManager, download_files
from .enums import FilterMethod, Platform, VerifyMethod
from .fetchers import fetch_global_android, fetch_japan_servers
from .filter import FileFilter
from .models import (
    DownloadResult,
    FileInfo,
    PackInfo,
    ResourceUnavailableError,
    TooManyFilesError,
)

logger = logging.getLogger(__name__)

_ALL_PLATFORMS = [Platform.GLOBAL_ANDROID, Platform.JAPAN_ANDROID, Platform.JAPAN_WINDOWS]


class BlueArchiveGameFilesDownloader:
    """Download Blue Archive game files across the three supported platforms.

    Args:
        data_dir: Fixed directory for the catalog DB (unless ``db_path`` is given)
            and the JP zip cache. Defaults to ``$BAGFD_DATA_DIR`` or
            ``platformdirs.user_data_dir("BAGFD")``.
        db_path: Optional override for the catalog DB's file path, independent of
            ``data_dir``. Defaults to ``$BAGFD_DB_PATH`` or ``data_dir/catalog.db``.
            Use this to keep the DB off storage that doesn't support the POSIX file
            locking SQLite's WAL mode relies on (e.g. a shared network volume),
            while still sharing ``data_dir`` for the zip cache.
        proxy: Optional HTTP/HTTPS proxy URL applied to all requests.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        db_path: Path | None = None,
        proxy: str | None = None,
    ):
        if data_dir is None:
            env = os.environ.get("BAGFD_DATA_DIR")
            if env:
                data_dir = Path(env)
            else:
                from platformdirs import user_data_dir
                data_dir = Path(user_data_dir("BAGFD"))
        self.data_dir = Path(data_dir)

        if db_path is None:
            env = os.environ.get("BAGFD_DB_PATH")
            db_path = Path(env) if env else self.data_dir / "catalog.db"
        self.db_path = Path(db_path)
        self.zip_cache = self.data_dir / "zip_cache"

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'x-cv': '3172501',
            'x-sv': '29',
            'x-abis': 'arm64-v8a,armeabi-v7a,armeabi',
            'x-gp': '1',
        })
        if proxy:
            self.session.proxies = {'http': proxy, 'https': proxy}

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        init_database(self.db_path)

        self._update_locks: dict[str, threading.Lock] = {
            p: threading.Lock() for p in _ALL_PLATFORMS
        }
        # Serializes access to shared cache files (the JP zip cache especially)
        # so multiple clients/threads don't open the same file at once.
        self._file_locks = PathLockManager(self.data_dir / "locks")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        pattern: str,
        platform: Platform,
        filter_method: FilterMethod = FilterMethod.AUTO,
        update_background: bool = False,
    ) -> list[FileInfo]:
        """Search the catalog for files matching ``pattern`` on ``platform``.

        Refreshes the catalog if stale, then returns metadata only — nothing is
        downloaded.

        Args:
            pattern: Filename pattern.
            platform: A single platform (``all`` is not accepted).
            filter_method: Matching strategy (see `FilterMethod`).
            update_background: If True, a due catalog refresh is kicked off on
                a daemon thread instead of blocking this call. The query then
                runs against whatever is currently in the catalog — possibly
                stale, or empty if this platform has never been fetched. At
                most one background refresh runs per platform at a time;
                redundant calls while one is in flight are skipped.

        Returns:
            A `FileInfo` per matching bundle file.

        Raises:
            ValueError: If ``platform`` is not one of the three platforms.
        """
        platform = self._validate_platform(platform)
        self._ensure_fresh(platform, background=update_background)
        f = FileFilter(pattern, filter_method)
        return self._query_platform(f, platform)

    def get_latest_files(
        self,
        pattern: str,
        platform: Platform,
        cache_dir: Path | None = None,
        verify: VerifyMethod = VerifyMethod.HASH,
        filter_method: FilterMethod = FilterMethod.AUTO,
        workers: int = 10,
        show_progress: bool = False,
        max_files: int | None = 50,
    ) -> list[Path]:
        """Ensure the latest matching files exist in ``cache_dir`` and return them.

        Downloads anything missing or stale and reuses what is already valid
        (per ``verify``), so the result is always the current version — whether
        freshly fetched or served from cache. For Japan, zip packs are cached in
        the shared zip cache and the matching members are extracted into
        ``cache_dir``.

        Args:
            pattern: Filename pattern.
            platform: A single platform (``all`` is not accepted).
            cache_dir: Where to store/return files. Defaults to
                ``data_dir/download_cache``.
            verify: Cache-reuse strategy (see `VerifyMethod`).
            filter_method: Matching strategy (see `FilterMethod`).
            workers: Parallel download workers.
            show_progress: Show a progress bar if tqdm is installed.
            max_files: Raise `TooManyFilesError` if more than this many match
                (default 50; pass ``None`` for unlimited). The guard keeps
                pipelines from accidentally pulling an entire platform.

        Returns:
            Paths to the matching files inside ``cache_dir``.

        Raises:
            ValueError: If ``platform`` is invalid.
            TooManyFilesError: If matches exceed ``max_files``.
        """
        platform = self._validate_platform(platform)
        cache_dir = Path(cache_dir) if cache_dir is not None else self.data_dir / "download_cache"
        self._ensure_fresh(platform, cache_dir=cache_dir)

        f = FileFilter(pattern, filter_method)
        matches = self._query_platform(f, platform)
        self._guard_count(matches, max_files)

        platform_cache = cache_dir / platform
        platform_cache.mkdir(parents=True, exist_ok=True)

        if platform == Platform.GLOBAL_ANDROID:
            items = [
                DownloadItem(
                    url=fi.url, dest=platform_cache / fi.name, size=fi.size,
                    hash_type=fi.hash_type, hash_value=fi.hash_value,
                )
                for fi in matches
            ]
            delivered = download_files(items, self.session, workers, show_progress, verify=verify, locks=self._file_locks)
            # download_files swallows per-file failures; a short delivery means
            # the server couldn't serve files the catalog lists — most likely a
            # maintenance window. Surface it as retryable rather than returning
            # a silently incomplete set.
            if len(delivered) < len(items):
                raise ResourceUnavailableError(
                    f"{len(items) - len(delivered)} of {len(items)} game file(s) for "
                    f"{platform} could not be downloaded — could not connect to the game "
                    f"server; try again later"
                )
            return delivered

        # Japan: fetch zip packs into the shared zip cache, extract matches.
        packs = self._group_japan(matches)
        zips_dir = self.zip_cache / platform
        self._fetch_japan_zips(packs, zips_dir, verify, workers, show_progress)
        # Any pack that didn't land means its bytes weren't served — treat the
        # same maintenance window as retryable instead of extracting a partial set.
        missing = [name for name in packs if not (zips_dir / name).exists()]
        if missing:
            raise ResourceUnavailableError(
                f"{len(missing)} of {len(packs)} game-file pack(s) for {platform} could "
                f"not be downloaded — could not connect to the game server; try again later"
            )
        return self._extract_japan(packs, zips_dir, platform_cache, with_path=False, overwrite=False)

    def download(
        self,
        pattern: str,
        platform: Platform,
        output_dir: str | Path = "./download",
        with_path: bool = False,
        verify: VerifyMethod = VerifyMethod.HASH,
        filter_method: FilterMethod = FilterMethod.AUTO,
        workers: int = 10,
        show_progress: bool = False,
        max_files: int | None = 50,
    ) -> DownloadResult:
        """Download the latest matching files into ``output_dir``.

        Files are written fresh into ``output_dir`` (Global bundles are always
        re-downloaded). For Japan, zip packs are fetched into the shared zip
        cache (reused when valid) and the matching members are extracted into
        ``output_dir``.

        Args:
            pattern: Filename pattern.
            platform: A single platform (``all`` is not accepted).
            output_dir: Destination directory. Defaults to ``./download``.
            with_path: If True, recreate each file's original relative path
                under ``output_dir``; if False (default), write files flat by
                their basename.
            verify: Cache-reuse strategy for the JP zip cache (see `VerifyMethod`).
            filter_method: Matching strategy (see `FilterMethod`).
            workers: Parallel download workers.
            show_progress: Show a progress bar if tqdm is installed.
            max_files: Raise `TooManyFilesError` if more than this many match
                (``None`` = unlimited).

        Returns:
            A `DownloadResult` with the delivered paths, count, and total bytes.

        Raises:
            ValueError: If ``platform`` is invalid.
            TooManyFilesError: If matches exceed ``max_files``.
        """
        platform = self._validate_platform(platform)
        self._ensure_fresh(platform)

        f = FileFilter(pattern, filter_method)
        matches = self._query_platform(f, platform)
        self._guard_count(matches, max_files)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if platform == Platform.GLOBAL_ANDROID:
            items = [
                DownloadItem(
                    url=fi.url,
                    dest=out / (fi.path if with_path else fi.name),
                    size=fi.size, hash_type=fi.hash_type, hash_value=fi.hash_value,
                )
                for fi in matches
            ]
            delivered = download_files(items, self.session, workers, show_progress, verify=verify, force=True, locks=self._file_locks)
        else:
            packs = self._group_japan(matches)
            zips_dir = self.zip_cache / platform
            self._fetch_japan_zips(packs, zips_dir, verify, workers, show_progress)
            delivered = self._extract_japan(packs, zips_dir, out, with_path=with_path, overwrite=True)

        total = sum(p.stat().st_size for p in delivered if p.exists())
        return DownloadResult(files=delivered, output_dir=out, total_bytes=total)

    def update(self, force: bool = False, platform="all", cache_dir: Path | None = None) -> None:
        """Refresh the file catalog for one or all platforms.

        When the catalog changes (new version, or a same-version content
        change such as a hotfix), stale caches for that platform are pruned:
        any cached file no longer in the catalog, or whose hash no longer
        matches, is removed; files still valid are kept.

        Args:
            force: Fetch even if the catalog was checked recently.
            platform: A platform, or ``"all"`` for every platform.
            cache_dir: Extra cache directory to invalidate on a new version
                (e.g. a custom ``get_latest_files`` cache).
        """
        for p in self._resolve_platforms(platform):
            if self._fetch_platform(p, force):
                self._prune_stale_cache(p, cache_dir)

    def clean(self, platform="all", cache_dir: Path | None = None) -> None:
        """Remove cached files and catalog rows for one or all platforms.

        Clears the zip cache, the default download cache, the catalog rows (and
        version record), and ``cache_dir`` if given.

        Args:
            platform: A platform, or ``"all"`` for every platform.
            cache_dir: Extra cache directory to clear too.
        """
        for p in self._resolve_platforms(platform):
            self._invalidate(p, cache_dir)
            clear_platform_db(self.db_path, p)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_platform(self, platform) -> str:
        """Return ``platform`` as a string, or raise if it isn't a real platform."""
        p = str(platform)
        if p not in _ALL_PLATFORMS:
            valid = ", ".join(pl.value for pl in Platform)
            raise ValueError(f"Invalid platform: {platform!r}. Use one of: {valid}")
        return p

    def _resolve_platforms(self, platform) -> list[str]:
        """Expand ``"all"`` / a single value / a list into a list of platforms."""
        if platform == "all":
            return list(_ALL_PLATFORMS)
        if isinstance(platform, str):
            return [platform]
        return list(platform)

    def _guard_count(self, matches: list, max_files: int | None) -> None:
        """Raise `TooManyFilesError` if ``matches`` exceeds ``max_files``."""
        if max_files is not None and len(matches) > max_files:
            raise TooManyFilesError(len(matches), max_files)

    def _fetch_platform(self, platform: str, force: bool) -> bool:
        """Refresh the catalog for one platform; return True if the catalog changed."""
        if platform == Platform.GLOBAL_ANDROID:
            return fetch_global_android(self.session, self.db_path, force)
        results = fetch_japan_servers(self.session, self.db_path, force)
        return bool(results.get(platform))

    def _ensure_fresh(self, platform: str, cache_dir: Path | None = None, background: bool = False) -> None:
        """Refresh the catalog if due, pruning caches on a catalog change."""
        if background:
            self._ensure_fresh_background(platform, cache_dir)
            return
        if self._fetch_platform(platform, force=False):
            self._prune_stale_cache(platform, cache_dir)

    def _ensure_fresh_background(self, platform: str, cache_dir: Path | None = None) -> None:
        """Refresh the catalog on a daemon thread; returns immediately.

        Skips starting a new thread if a refresh for ``platform`` is already
        in flight, so repeated calls don't pile up redundant fetches.
        """
        lock = self._update_locks[platform]
        if not lock.acquire(blocking=False):
            return

        def run() -> None:
            try:
                if self._fetch_platform(platform, force=False):
                    self._prune_stale_cache(platform, cache_dir)
            except Exception:
                logger.exception(f"Background catalog update failed for {platform}")
            finally:
                lock.release()

        threading.Thread(target=run, name=f"bagfd-update-{platform}", daemon=True).start()

    def _invalidate(self, platform: str, cache_dir: Path | None = None) -> None:
        """Clear cached files for ``platform`` across all cache locations.

        Blanket clear — used by `clean()`, which drops the catalog rows right
        after, so there's nothing to diff against. Catalog-refresh paths use
        `_prune_stale_cache` instead, which keeps files still valid.
        """
        clear_cache_for_platform(self.zip_cache, platform, self._file_locks)
        clear_cache_for_platform(self.data_dir / "download_cache", platform, self._file_locks)
        if cache_dir is not None:
            clear_cache_for_platform(Path(cache_dir), platform, self._file_locks)

    def _valid_hashes(self, table_name: str, basename: bool) -> dict[str, tuple[str, str]]:
        """Map cache filename -> (hash_type, hash_value) from current catalog rows.

        Args:
            table_name: Catalog table to read.
            basename: Global's catalog `path` includes a directory prefix
                (e.g. ``Android/aa/xyz.bundle``) but cache files are named by
                basename only; Japan's `path` (the pack name) already matches
                the cache filename directly.
        """
        rows = get_game_files(self.db_path, table_name)
        if basename:
            return {path.split('/')[-1]: (hash_type, hash_value) for path, _url, hash_type, hash_value, _size, _bf in rows}
        return {path: (hash_type, hash_value) for path, _url, hash_type, hash_value, _size, _bf in rows}

    def _prune_stale_cache(self, platform: str, cache_dir: Path | None = None) -> None:
        """Prune caches after a catalog change, keeping files still valid.

        Global / Japan zip packs are hash-verified against the current catalog
        row for their path — a file whose hash still matches survives. Japan's
        extracted individual files aren't catalog-tracked on their own (only
        the pack they came from is), and re-extracting from an already-valid
        zip is cheap, so that tier is cleared outright instead.
        """
        table_name = get_table_name(platform)
        if platform == Platform.GLOBAL_ANDROID:
            valid = self._valid_hashes(table_name, basename=True)
            prune_stale_cache(self.data_dir / "download_cache", platform, valid, self._file_locks)
            if cache_dir is not None:
                prune_stale_cache(Path(cache_dir), platform, valid, self._file_locks)
            return

        valid = self._valid_hashes(table_name, basename=False)
        prune_stale_cache(self.zip_cache, platform, valid, self._file_locks)
        clear_cache_for_platform(self.data_dir / "download_cache", platform, self._file_locks)
        if cache_dir is not None:
            clear_cache_for_platform(Path(cache_dir), platform, self._file_locks)

    def _query_platform(self, f: FileFilter, platform: str) -> list[FileInfo]:
        """Match ``f`` against the catalog rows for ``platform``."""
        rows = get_game_files(self.db_path, get_table_name(platform))
        result: list[FileInfo] = []
        if platform == Platform.GLOBAL_ANDROID:
            for path, url, hash_type, hash_value, size, _bundle in rows:
                name = path.split('/')[-1]
                if f.matches(name):
                    result.append(FileInfo(
                        name=name, platform=platform, path=path, url=url,
                        hash_type=hash_type, hash_value=hash_value, size=size, pack=None,
                    ))
        else:
            for pack_name, url, hash_type, hash_value, pack_size, bundle_files_json in rows:
                bundle_files = json.loads(bundle_files_json) if bundle_files_json else []
                matched = [bf for bf in bundle_files if f.matches(bf)]
                if not matched:
                    continue
                pack = PackInfo(
                    name=pack_name, url=url, hash_type=hash_type,
                    hash_value=hash_value, size=pack_size, files=bundle_files,
                )
                for bf in matched:
                    result.append(FileInfo(
                        name=bf, platform=platform, path=None, url=None,
                        hash_type=None, hash_value=None, size=None, pack=pack,
                    ))
        return result

    def _group_japan(self, matches: list[FileInfo]) -> dict[str, tuple[PackInfo, set[str]]]:
        """Group matched Japan files by the single pack to fetch each from.

        The same bundle can appear in several packs (e.g. a FullPatch and an
        UpdatePatch). Since the filename embeds a content hash, identical names
        mean identical bytes — so each bundle is assigned to exactly one pack,
        preferring the smallest pack that contains it. This avoids downloading a
        large pack for a file already covered by a small one, and prevents
        duplicate paths in the result.

        Returns a map of ``pack_name -> (PackInfo, {matched member names})``.
        """
        packs: dict[str, tuple[PackInfo, set[str]]] = {}
        seen: set[str] = set()
        for fi in sorted(matches, key=lambda f: f.pack.size):
            if fi.name in seen:
                continue
            seen.add(fi.name)
            packs.setdefault(fi.pack.name, (fi.pack, set()))[1].add(fi.name)
        return packs

    def _fetch_japan_zips(
        self,
        packs: dict[str, tuple[PackInfo, set[str]]],
        zips_dir: Path,
        verify: VerifyMethod,
        workers: int,
        show_progress: bool,
    ) -> None:
        """Ensure the zip pack for each entry in ``packs`` exists in ``zips_dir``."""
        if not packs:
            return
        zips_dir.mkdir(parents=True, exist_ok=True)
        items = [
            DownloadItem(
                url=pk.url, dest=zips_dir / name, size=pk.size,
                hash_type=pk.hash_type, hash_value=pk.hash_value,
            )
            for name, (pk, _matched) in packs.items()
        ]
        download_files(items, self.session, workers, show_progress, verify=verify, locks=self._file_locks)

    def _extract_japan(
        self,
        packs: dict[str, tuple[PackInfo, set[str]]],
        zips_dir: Path,
        dest_dir: Path,
        with_path: bool,
        overwrite: bool,
    ) -> list[Path]:
        """Extract matched members from cached zips into ``dest_dir``.

        Args:
            packs: Output of `_group_japan`.
            zips_dir: Directory holding the cached zip packs.
            dest_dir: Where extracted members are written.
            with_path: Keep each member's path inside the zip; otherwise flatten
                to its basename.
            overwrite: Re-extract even if the target already exists.

        Returns:
            Paths of the extracted (or already-present) member files.
        """
        dest_root = dest_dir.resolve()
        delivered: list[Path] = []
        for name, (_pk, matched) in packs.items():
            zip_path = zips_dir / name
            if not zip_path.exists():
                continue
            # Hold the same per-file lock used for downloading so we never read a
            # zip another client/thread is still writing (the WinError 32 path).
            with self._file_locks.lock(zip_path), zipfile.ZipFile(zip_path, 'r') as zf:
                for member in zf.namelist():
                    if member not in matched:
                        continue
                    rel = member if with_path else Path(member).name
                    target = dest_dir / rel
                    try:
                        if not target.resolve().is_relative_to(dest_root):
                            logger.warning(f"Blocked path traversal attempt: {member}")
                            continue
                    except ValueError:
                        logger.warning(f"Blocked path traversal attempt: {member}")
                        continue
                    if not overwrite and target.exists():
                        delivered.append(target)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, 'wb') as dst:
                        dst.write(src.read())
                    delivered.append(target)
        return delivered
