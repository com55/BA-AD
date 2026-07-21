"""
Unit tests for bagfd.downloader — no network required.
"""
import hashlib
import threading
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

from bagfd.downloader import DownloadItem, PathLockManager, _hash_matches, _is_valid_cache
from bagfd.enums import VerifyMethod


def _proc_lock_worker(args):
    """Acquire the path lock in a separate process and record the held window.

    Must be module-level so it is picklable for ProcessPoolExecutor.
    """
    lock_dir, target, hold = args
    mgr = PathLockManager(lock_dir, timeout=30)
    # Wall-clock time.time() is comparable across processes; monotonic is not.
    with mgr.lock(target):
        start = time.time()
        time.sleep(hold)
        return (start, time.time())


class TestHashMatches:
    def test_md5_match(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello")
        assert _hash_matches(f, "md5", hashlib.md5(b"hello").hexdigest())

    def test_md5_case_insensitive(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello")
        assert _hash_matches(f, "MD5", hashlib.md5(b"hello").hexdigest().upper())

    def test_md5_mismatch(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello")
        assert not _hash_matches(f, "md5", "deadbeef")

    def test_crc32_match_unsigned(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello")
        assert _hash_matches(f, "crc32", str(zlib.crc32(b"hello") & 0xFFFFFFFF))

    def test_crc32_match_signed(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello")
        signed = (zlib.crc32(b"hello") & 0xFFFFFFFF) - 2**32
        assert _hash_matches(f, "crc32", str(signed))

    def test_unknown_hash_type(self, tmp_path):
        f = tmp_path / "a.bin"
        f.write_bytes(b"hello")
        assert not _hash_matches(f, "sha256", "whatever")


class TestIsValidCache:
    def _item(self, tmp_path, data=b"hello"):
        f = tmp_path / "a.bin"
        f.write_bytes(data)
        return DownloadItem(
            url="http://x", dest=f, size=len(data),
            hash_type="md5", hash_value=hashlib.md5(data).hexdigest(),
        )

    def test_missing_file_never_valid(self, tmp_path):
        item = DownloadItem(url="http://x", dest=tmp_path / "missing", size=1)
        assert not _is_valid_cache(item, VerifyMethod.NONE)

    def test_none_reuses_if_exists(self, tmp_path):
        assert _is_valid_cache(self._item(tmp_path), VerifyMethod.NONE)

    def test_size_match(self, tmp_path):
        assert _is_valid_cache(self._item(tmp_path), VerifyMethod.SIZE)

    def test_size_mismatch(self, tmp_path):
        item = self._item(tmp_path)
        item.size = 999
        assert not _is_valid_cache(item, VerifyMethod.SIZE)

    def test_hash_match(self, tmp_path):
        assert _is_valid_cache(self._item(tmp_path), VerifyMethod.HASH)

    def test_hash_mismatch(self, tmp_path):
        item = self._item(tmp_path)
        item.hash_value = "deadbeef"
        assert not _is_valid_cache(item, VerifyMethod.HASH)

    def test_hash_falls_back_to_size_when_no_hash(self, tmp_path):
        item = self._item(tmp_path)
        item.hash_type = None
        item.hash_value = None
        assert _is_valid_cache(item, VerifyMethod.HASH)  # size still matches


class TestPathLockManager:
    """The lock must serialize access to one path — across threads and processes.

    WinError 32 is a Windows-only symptom and can't be reproduced on Linux, so
    these tests instead verify the mechanism the fix relies on: that only one
    holder is ever inside the critical section for a given path at a time.
    """

    def test_serializes_threads_on_same_path(self, tmp_path):
        mgr = PathLockManager(tmp_path / "locks")
        target = tmp_path / "cache" / "pack.zip"
        state = {"current": 0, "max": 0}

        def worker():
            with mgr.lock(target):
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
                time.sleep(0.05)
                state["current"] -= 1

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["max"] == 1

    def test_different_paths_run_concurrently(self, tmp_path):
        """Distinct paths must not block each other (the lock is per-path)."""
        mgr = PathLockManager(tmp_path / "locks")
        state = {"current": 0, "max": 0}
        guard = threading.Lock()

        def worker(i):
            with mgr.lock(tmp_path / f"pack_{i}.zip"):
                with guard:
                    state["current"] += 1
                    state["max"] = max(state["max"], state["current"])
                time.sleep(0.1)
                with guard:
                    state["current"] -= 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert state["max"] > 1

    def test_serializes_separate_processes_on_same_path(self, tmp_path):
        lock_dir = str(tmp_path / "locks")
        target = str(tmp_path / "cache" / "pack.zip")
        hold = 0.3
        args = [(lock_dir, target, hold)] * 2
        with ProcessPoolExecutor(max_workers=2) as ex:
            windows = list(ex.map(_proc_lock_worker, args))

        windows.sort()
        # Disjoint held windows: the second process started only after the first
        # released. A tiny tolerance absorbs wall-clock jitter between procs.
        assert windows[1][0] >= windows[0][1] - 0.01


class TestInvalidationSkipsInUseFiles:
    """clear_cache_for_platform must not delete a file another holder is using."""

    def test_skips_locked_file_keeps_it(self, tmp_path):
        from bagfd.database import clear_cache_for_platform

        mgr = PathLockManager(tmp_path / "locks")
        plat_dir = tmp_path / "zip_cache" / "japan-android"
        plat_dir.mkdir(parents=True)
        busy = plat_dir / "Pack_busy.zip"
        free = plat_dir / "Pack_free.zip"
        busy.write_bytes(b"BUSY")
        free.write_bytes(b"FREE")

        # Hold the busy file's lock while invalidation runs (simulates a
        # concurrent download/extract). It must be skipped; the free one goes.
        with mgr.lock(busy):
            clear_cache_for_platform(tmp_path / "zip_cache", "japan-android", mgr)

        assert busy.exists()       # in-use file preserved, not deleted/crashed
        assert not free.exists()   # idle file still cleared

    def test_removes_all_when_unlocked(self, tmp_path):
        from bagfd.database import clear_cache_for_platform

        mgr = PathLockManager(tmp_path / "locks")
        plat_dir = tmp_path / "zip_cache" / "japan-android"
        plat_dir.mkdir(parents=True)
        (plat_dir / "a.zip").write_bytes(b"A")
        (plat_dir / "b.zip").write_bytes(b"B")

        clear_cache_for_platform(tmp_path / "zip_cache", "japan-android", mgr)

        assert list(plat_dir.iterdir()) == []


class TestPruneStaleCache:
    """prune_stale_cache must keep files whose catalog entry+hash still match
    and evict everything else (dropped from the catalog, or hash-mismatched)."""

    def test_keeps_valid_removes_mismatch_and_orphan(self, tmp_path):
        from bagfd.database import prune_stale_cache

        plat_dir = tmp_path / "cache" / "global-android"
        plat_dir.mkdir(parents=True)
        valid = plat_dir / "valid.bundle"
        valid.write_bytes(b"KEEP")
        stale_hash = plat_dir / "stale_hash.bundle"
        stale_hash.write_bytes(b"OLD")
        orphan = plat_dir / "orphan.bundle"
        orphan.write_bytes(b"GONE")

        valid_hashes = {
            "valid.bundle": ("md5", hashlib.md5(b"KEEP").hexdigest()),
            "stale_hash.bundle": ("md5", hashlib.md5(b"NEW").hexdigest()),  # on-disk content is stale
            # orphan.bundle intentionally has no catalog row
        }

        prune_stale_cache(tmp_path / "cache", "global-android", valid_hashes)

        assert valid.exists()
        assert not stale_hash.exists()
        assert not orphan.exists()

    def test_skips_locked_file_keeps_it(self, tmp_path):
        from bagfd.database import prune_stale_cache

        mgr = PathLockManager(tmp_path / "locks")
        plat_dir = tmp_path / "cache" / "global-android"
        plat_dir.mkdir(parents=True)
        busy = plat_dir / "busy.bundle"
        busy.write_bytes(b"BUSY")

        # No catalog entry -> would normally be pruned, but it's held locked
        # (simulates a concurrent downloader/extractor).
        with mgr.lock(busy):
            prune_stale_cache(tmp_path / "cache", "global-android", {}, mgr)

        assert busy.exists()
