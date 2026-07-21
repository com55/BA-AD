"""
Unit tests for bagfd.client (BlueArchiveGameFilesDownloader) — no network required.
"""
import hashlib
import json
import sqlite3
import threading
import zipfile
import zlib
from unittest.mock import patch

import pytest

from bagfd import (
    BlueArchiveGameFilesDownloader,
    ResourceUnavailableError,
    TooManyFilesError,
)
from bagfd.database import (
    get_stored_version,
    get_table_name,
    save_game_files,
    update_version,
)
from bagfd.enums import VerifyMethod


# ---------------------------------------------------------------------------
# Constructor / data_dir resolution
# ---------------------------------------------------------------------------

class TestDataDir:
    def test_explicit_arg(self, tmp_path):
        d = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        assert d.data_dir == tmp_path
        assert d.db_path == tmp_path / "catalog.db"
        assert d.zip_cache == tmp_path / "zip_cache"

    def test_explicit_arg_creates_directory(self, tmp_path):
        target = tmp_path / "nested" / "data"
        BlueArchiveGameFilesDownloader(data_dir=target)
        assert target.exists()

    def test_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BAGFD_DATA_DIR", str(tmp_path))
        d = BlueArchiveGameFilesDownloader()
        assert d.data_dir == tmp_path

    def test_arg_overrides_env_var(self, tmp_path, tmp_path_factory, monkeypatch):
        env_dir = tmp_path_factory.mktemp("env")
        monkeypatch.setenv("BAGFD_DATA_DIR", str(env_dir))
        d = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        assert d.data_dir == tmp_path

    def test_default_uses_platformdirs(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BAGFD_DATA_DIR", raising=False)
        with patch("platformdirs.user_data_dir", return_value=str(tmp_path)) as mock_udd:
            d = BlueArchiveGameFilesDownloader()
        mock_udd.assert_called_once_with("BAGFD")
        assert d.data_dir == tmp_path

    def test_db_created_in_data_dir(self, tmp_path):
        BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        assert (tmp_path / "catalog.db").exists()

    def test_db_path_overrides_data_dir(self, tmp_path):
        db_dir = tmp_path / "db_elsewhere"
        db_path = db_dir / "catalog.db"
        d = BlueArchiveGameFilesDownloader(data_dir=tmp_path, db_path=db_path)
        assert d.db_path == db_path
        assert db_path.exists()
        assert not (tmp_path / "catalog.db").exists()
        assert d.zip_cache == tmp_path / "zip_cache"

    def test_db_path_env_var(self, tmp_path, monkeypatch):
        db_path = tmp_path / "db_elsewhere" / "catalog.db"
        monkeypatch.setenv("BAGFD_DB_PATH", str(db_path))
        d = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        assert d.db_path == db_path
        assert db_path.exists()

    def test_db_path_arg_overrides_env_var(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env" / "catalog.db"
        arg_path = tmp_path / "arg" / "catalog.db"
        monkeypatch.setenv("BAGFD_DB_PATH", str(env_path))
        d = BlueArchiveGameFilesDownloader(data_dir=tmp_path, db_path=arg_path)
        assert d.db_path == arg_path
        assert not env_path.exists()

    def test_proxy_sets_session_proxies(self, tmp_path):
        d = BlueArchiveGameFilesDownloader(data_dir=tmp_path, proxy="http://proxy:8080")
        assert d.session.proxies == {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}

    def test_no_proxy_leaves_session_unset(self, tmp_path):
        d = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        assert not d.session.proxies


# ---------------------------------------------------------------------------
# _resolve_platforms
# ---------------------------------------------------------------------------

class TestResolvePlatforms:
    @pytest.fixture
    def client(self, tmp_path):
        return BlueArchiveGameFilesDownloader(data_dir=tmp_path)

    def test_all(self, client):
        assert client._resolve_platforms('all') == ['global-android', 'japan-android', 'japan-windows']

    def test_single_string(self, client):
        assert client._resolve_platforms('global-android') == ['global-android']

    def test_list(self, client):
        assert client._resolve_platforms(['global-android', 'japan-android']) == ['global-android', 'japan-android']


# ---------------------------------------------------------------------------
# Helpers: seed test data into DB
# ---------------------------------------------------------------------------

def _seed_versions(db_path) -> None:
    """Mark all platforms as recently checked so _ensure_fresh() skips network."""
    for platform in ['global-android', 'japan-android', 'japan-windows']:
        update_version(db_path, platform, '1.0.0')


def _seed_global(db_path) -> None:
    table = get_table_name("global-android")
    files = [
        ("Android/ch0230_foo.bundle", "https://cdn/ch0230_foo.bundle", "md5", "abc", 1024, None),
        ("Android/ch0231_bar.bundle", "https://cdn/ch0231_bar.bundle", "md5", "def", 2048, None),
        ("Android/Image_CueSheet_001.bundle", "https://cdn/Image_CueSheet_001.bundle", "md5", "ghi", 512, None),
    ]
    save_game_files(db_path, table, files)


def _seed_japan(db_path, platform: str = "japan-android") -> None:
    table = get_table_name(platform)
    files = [
        (
            "Pack_ch0230.zip",
            "https://jp/Pack_ch0230.zip",
            "crc32", "111", 8192,
            json.dumps(["ch0230_a.bundle", "ch0230_b.bundle"]),
        ),
        (
            "Pack_Image.zip",
            "https://jp/Pack_Image.zip",
            "crc32", "222", 4096,
            json.dumps(["Image_CueSheet_001.bundle", "Image_CueSheet_002.bundle"]),
        ),
    ]
    save_game_files(db_path, table, files)


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestQuery:
    @pytest.fixture
    def client(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_global(c.db_path)
        _seed_japan(c.db_path, "japan-android")
        _seed_versions(c.db_path)
        return c

    def test_global_contains_match(self, client):
        results = client.query('ch0230', platform='global-android')
        assert len(results) == 1
        fi = results[0]
        assert fi.name == 'ch0230_foo.bundle'
        assert fi.platform == 'global-android'
        assert fi.size == 1024
        assert fi.pack is None

    def test_global_fileinfo_fields(self, client):
        fi = client.query('ch0230', platform='global-android')[0]
        assert fi.path == 'Android/ch0230_foo.bundle'
        assert fi.url == 'https://cdn/ch0230_foo.bundle'
        assert fi.hash_type == 'md5'
        assert fi.hash_value == 'abc'

    def test_global_glob_match(self, client):
        results = client.query('*.bundle', platform='global-android')
        assert len(results) == 3

    def test_global_no_match(self, client):
        assert client.query('nonexistent', platform='global-android') == []

    def test_japan_contains_match(self, client):
        results = client.query('ch0230', platform='japan-android')
        assert len(results) == 2
        names = {fi.name for fi in results}
        assert names == {'ch0230_a.bundle', 'ch0230_b.bundle'}

    def test_japan_fileinfo_has_pack(self, client):
        results = client.query('ch0230_a.bundle', platform='japan-android', filter_method='contains')
        assert len(results) == 1
        fi = results[0]
        # per-file fields are empty on Japan; the data lives on the pack
        assert fi.path is None
        assert fi.url is None
        assert fi.hash_type is None
        assert fi.size is None
        assert fi.pack is not None
        assert fi.pack.name == 'Pack_ch0230.zip'
        assert fi.pack.url == 'https://jp/Pack_ch0230.zip'
        assert fi.pack.hash_type == 'crc32'
        assert fi.pack.hash_value == '111'
        assert fi.pack.size == 8192

    def test_japan_pack_files_are_names(self, client):
        results = client.query('ch0230_a.bundle', platform='japan-android', filter_method='contains')
        pack = results[0].pack
        # files is the full member list of the zip, as plain strings
        assert pack.files == ['ch0230_a.bundle', 'ch0230_b.bundle']

    def test_rejects_all_platform(self, client):
        with pytest.raises(ValueError):
            client.query('ch0230', platform='all')

    def test_rejects_unknown_platform(self, client):
        with pytest.raises(ValueError):
            client.query('ch0230', platform='switch')


# ---------------------------------------------------------------------------
# query() — update_background
# ---------------------------------------------------------------------------

class TestQueryUpdateBackground:
    def test_returns_immediately_without_blocking(self, tmp_path):
        client = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_global(client.db_path)
        started = threading.Event()
        release = threading.Event()

        def slow_fetch(session, db_path, force=False, check_interval=None):
            started.set()
            release.wait(timeout=2)
            return False

        with patch('bagfd.client.fetch_global_android', side_effect=slow_fetch):
            results = client.query('ch0230', platform='global-android', update_background=True)
            assert started.wait(timeout=1), "background fetch never started"
        assert len(results) == 1
        release.set()

    def test_dedupes_concurrent_background_calls(self, tmp_path):
        client = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_global(client.db_path)
        release = threading.Event()
        calls = []

        def slow_fetch(session, db_path, force=False, check_interval=None):
            calls.append(1)
            release.wait(timeout=2)
            return False

        with patch('bagfd.client.fetch_global_android', side_effect=slow_fetch):
            client.query('ch0230', platform='global-android', update_background=True)
            client.query('ch0230', platform='global-android', update_background=True)
            release.set()
            lock = client._update_locks['global-android']
            assert lock.acquire(timeout=2)
            lock.release()

        assert len(calls) == 1

    def test_background_thread_populates_empty_catalog(self, tmp_path):
        client = BlueArchiveGameFilesDownloader(data_dir=tmp_path)

        def fake_fetch(session, db_path, force=False, check_interval=None):
            _seed_global(db_path)
            update_version(db_path, 'global-android', '1.0.0', is_new_version=True)
            return True

        with patch('bagfd.client.fetch_global_android', side_effect=fake_fetch):
            results = client.query('ch0230', platform='global-android', update_background=True)
            assert results == []

            lock = client._update_locks['global-android']
            assert lock.acquire(timeout=2)
            lock.release()

            results = client.query('ch0230', platform='global-android')
        assert len(results) == 1


# ---------------------------------------------------------------------------
# download() — guard
# ---------------------------------------------------------------------------

class TestDownloadGuard:
    @pytest.fixture
    def client_51(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        table = get_table_name("global-android")
        files = [
            (f"Android/f{i:03d}.bundle", f"https://cdn/f{i:03d}.bundle", "md5", "x", 1024, None)
            for i in range(51)
        ]
        save_game_files(c.db_path, table, files)
        _seed_versions(c.db_path)
        return c

    def test_raises_over_default_limit(self, client_51, tmp_path):
        with pytest.raises(TooManyFilesError) as exc:
            client_51.download('*.bundle', platform='global-android', output_dir=tmp_path / 'out')
        assert exc.value.count == 51
        assert exc.value.limit == 50

    def test_max_files_none_bypasses(self, client_51, tmp_path):
        with patch('bagfd.client.download_files', return_value=[]):
            client_51.download('*.bundle', platform='global-android',
                               output_dir=tmp_path / 'out', max_files=None)

    def test_within_limit_passes(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_global(c.db_path)   # 3 files
        _seed_versions(c.db_path)
        with patch('bagfd.client.download_files', return_value=[]):
            c.download('*.bundle', platform='global-android', output_dir=tmp_path / 'out')

    def test_custom_limit(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_global(c.db_path)   # 3 files
        _seed_versions(c.db_path)
        with pytest.raises(TooManyFilesError) as exc:
            c.download('*.bundle', platform='global-android',
                       output_dir=tmp_path / 'out', max_files=2)
        assert exc.value.count == 3
        assert exc.value.limit == 2

    def test_get_latest_files_default_limit(self, client_51, tmp_path):
        with pytest.raises(TooManyFilesError) as exc:
            client_51.get_latest_files('*.bundle', platform='global-android',
                                       cache_dir=tmp_path / 'c')
        assert exc.value.count == 51
        assert exc.value.limit == 50

    def test_get_latest_files_none_bypasses(self, client_51, tmp_path):
        # Return a full delivery (51 paths) so the completeness guard is
        # satisfied — this test only asserts max_files=None bypasses the limit.
        delivered = [tmp_path / 'c' / f'f{i:03d}.bundle' for i in range(51)]
        with patch('bagfd.client.download_files', return_value=delivered):
            client_51.get_latest_files('*.bundle', platform='global-android',
                                       cache_dir=tmp_path / 'c', max_files=None)


# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------

class TestClean:
    @pytest.fixture
    def client(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_global(c.db_path)
        _seed_japan(c.db_path, "japan-android")
        ga_dir = c.data_dir / 'download_cache' / 'global-android'
        ga_dir.mkdir(parents=True)
        (ga_dir / 'ch0230_foo.bundle').write_bytes(b'data')
        zip_dir = c.zip_cache / 'global-android'
        zip_dir.mkdir(parents=True)
        (zip_dir / 'Pack_x.zip').write_bytes(b'zip')
        return c

    def test_clean_removes_cached_files_but_keeps_dir(self, client):
        ga_dir = client.data_dir / 'download_cache' / 'global-android'
        client.clean(platform='global-android')
        assert ga_dir.exists()                              # folder kept
        assert not (ga_dir / 'ch0230_foo.bundle').exists()  # contents gone

    def test_clean_clears_zip_cache_contents(self, client):
        zip_dir = client.zip_cache / 'global-android'
        client.clean(platform='global-android')
        assert zip_dir.exists()
        assert not (zip_dir / 'Pack_x.zip').exists()

    def test_clean_leaves_cache_dir_and_siblings(self, client):
        cache_root = client.data_dir / 'download_cache'
        (cache_root / 'japan-android').mkdir(parents=True)
        (cache_root / 'japan-android' / 'keep.bundle').write_bytes(b'keep')
        client.clean(platform='global-android')
        assert cache_root.exists()                                   # parent untouched
        assert (cache_root / 'japan-android' / 'keep.bundle').exists()  # sibling untouched

    def test_clean_removes_db_entries(self, client):
        client.clean(platform='global-android')
        conn = sqlite3.connect(client.db_path)
        rows = conn.execute("SELECT COUNT(*) FROM global_android").fetchone()[0]
        conn.close()
        assert rows == 0

    def test_clean_removes_version(self, client):
        update_version(client.db_path, 'global-android', '1.0.0')
        client.clean(platform='global-android')
        assert get_stored_version(client.db_path, 'global-android') is None

    def test_clean_all_clears_all_platforms(self, client):
        client.clean(platform='all')
        for platform in ['global-android', 'japan-android', 'japan-windows']:
            conn = sqlite3.connect(client.db_path)
            rows = conn.execute(f"SELECT COUNT(*) FROM {get_table_name(platform)}").fetchone()[0]
            conn.close()
            assert rows == 0

    def test_clean_japan_leaves_global(self, client):
        client.clean(platform='japan-android')
        conn = sqlite3.connect(client.db_path)
        rows = conn.execute("SELECT COUNT(*) FROM global_android").fetchone()[0]
        conn.close()
        assert rows == 3


# ---------------------------------------------------------------------------
# update() — cache pruning on a catalog change
# ---------------------------------------------------------------------------

class TestUpdatePrunesCache:
    """update() must keep cached files still valid after a catalog refresh,
    evicting only what changed or dropped out — not a blanket wipe."""

    def test_global_keeps_valid_evicts_stale(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_versions(c.db_path)

        cache_dir = c.data_dir / "download_cache" / "global-android"
        cache_dir.mkdir(parents=True)
        (cache_dir / "keep.bundle").write_bytes(b"KEEP")
        (cache_dir / "stale.bundle").write_bytes(b"STALE")

        def fake_fetch(session, db_path, force=False, check_interval=None):
            save_game_files(db_path, get_table_name("global-android"), [
                ("Android/keep.bundle", "https://cdn/keep.bundle", "md5", hashlib.md5(b"KEEP").hexdigest(), 4, None),
                # stale.bundle's catalog row is gone entirely (dropped from the catalog)
            ])
            return True

        with patch('bagfd.client.fetch_global_android', side_effect=fake_fetch):
            c.update(platform='global-android')

        assert (cache_dir / "keep.bundle").exists()
        assert not (cache_dir / "stale.bundle").exists()

    def test_japan_hash_sweeps_zip_cache_but_clears_extracted_files(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_versions(c.db_path)

        zip_dir = c.zip_cache / "japan-android"
        zip_dir.mkdir(parents=True)
        (zip_dir / "Pack_keep.zip").write_bytes(b"KEEP")
        (zip_dir / "Pack_stale.zip").write_bytes(b"STALE")

        extracted_dir = c.data_dir / "download_cache" / "japan-android"
        extracted_dir.mkdir(parents=True)
        (extracted_dir / "member.bundle").write_bytes(b"whatever")

        def fake_fetch(session, db_path, force=False, check_interval=None):
            save_game_files(db_path, get_table_name("japan-android"), [
                ("Pack_keep.zip", "https://jp/Pack_keep.zip", "crc32",
                 str(zlib.crc32(b"KEEP") & 0xFFFFFFFF), 4, "[]"),
                # Pack_stale.zip's catalog row is gone entirely
            ])
            return {"japan-android": True, "japan-windows": False}

        with patch('bagfd.client.fetch_japan_servers', side_effect=fake_fetch):
            c.update(platform='japan-android')

        assert (zip_dir / "Pack_keep.zip").exists()          # hash-verified, still valid
        assert not (zip_dir / "Pack_stale.zip").exists()     # gone from catalog, evicted
        # Extracted individual files aren't catalog-tracked on their own;
        # cheaper to just clear them and let extraction from the (now-valid)
        # zip cache repopulate them.
        assert not (extracted_dir / "member.bundle").exists()

    def test_no_catalog_change_leaves_cache_untouched(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path)
        _seed_versions(c.db_path)

        cache_dir = c.data_dir / "download_cache" / "global-android"
        cache_dir.mkdir(parents=True)
        (cache_dir / "untouched.bundle").write_bytes(b"DATA")

        with patch('bagfd.client.fetch_global_android', return_value=False):
            c.update(platform='global-android')

        assert (cache_dir / "untouched.bundle").exists()


# ---------------------------------------------------------------------------
# download() / get_latest_files() — delivery (network mocked)
# ---------------------------------------------------------------------------

def _fake_download(items, session, workers=10, show_progress=False, verify=VerifyMethod.HASH, force=False, locks=None):
    """Stand-in for download_files: writes a stub file at each destination."""
    out = []
    for it in items:
        it.dest.parent.mkdir(parents=True, exist_ok=True)
        it.dest.write_bytes(b'XXXX')
        out.append(it.dest)
    return out


class TestDownloadDelivery:
    @pytest.fixture
    def client(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path / "data")
        _seed_global(c.db_path)
        _seed_japan(c.db_path, "japan-android")
        _seed_versions(c.db_path)
        return c

    def test_global_flat(self, client, tmp_path):
        out = tmp_path / "out"
        with patch('bagfd.client.download_files', side_effect=_fake_download):
            result = client.download('*.bundle', platform='global-android', output_dir=out, max_files=None)
        assert result.count == 3
        assert result.total_bytes == 12  # 3 files * 4 bytes
        assert (out / 'ch0230_foo.bundle').exists()
        assert result.output_dir == out

    def test_global_with_path(self, client, tmp_path):
        out = tmp_path / "out"
        with patch('bagfd.client.download_files', side_effect=_fake_download):
            client.download('ch0230', platform='global-android', output_dir=out, with_path=True)
        assert (out / 'Android' / 'ch0230_foo.bundle').exists()

    def test_japan_extracts_matching_member_to_output(self, client, tmp_path):
        # Pre-place the zip so the (patched, no-op) downloader doesn't hit network.
        zpath = client.zip_cache / 'japan-android' / 'Pack_ch0230.zip'
        zpath.parent.mkdir(parents=True)
        with zipfile.ZipFile(zpath, 'w') as zf:
            zf.writestr('ch0230_a.bundle', b'AAA')
            zf.writestr('ch0230_b.bundle', b'BBB')

        out = tmp_path / "out"
        with patch('bagfd.client.download_files', return_value=[]):
            result = client.download('ch0230_a.bundle', platform='japan-android',
                                     output_dir=out, filter_method='contains')
        assert result.count == 1
        assert (out / 'ch0230_a.bundle').read_bytes() == b'AAA'
        assert not (out / 'ch0230_b.bundle').exists()  # not matched

    def test_get_latest_files_global_returns_cache_paths(self, client, tmp_path):
        cache = tmp_path / "mycache"
        with patch('bagfd.client.download_files', side_effect=_fake_download):
            paths = client.get_latest_files('ch0230', platform='global-android', cache_dir=cache)
        assert paths == [cache / 'global-android' / 'ch0230_foo.bundle']
        assert paths[0].exists()

    def test_japan_dedups_to_smallest_pack(self, tmp_path):
        # same bundle present in two packs of different size -> use the smaller
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path / "data")
        table = get_table_name("japan-android")
        save_game_files(c.db_path, table, [
            ("Big.zip",   "https://jp/Big.zip",   "crc32", "1", 9999, json.dumps(["shared.bundle"])),
            ("Small.zip", "https://jp/Small.zip", "crc32", "2", 10,   json.dumps(["shared.bundle"])),
        ])
        _seed_versions(c.db_path)
        zdir = c.zip_cache / "japan-android"
        zdir.mkdir(parents=True)
        with zipfile.ZipFile(zdir / "Big.zip", "w") as zf:
            zf.writestr("shared.bundle", b"BIG")
        with zipfile.ZipFile(zdir / "Small.zip", "w") as zf:
            zf.writestr("shared.bundle", b"SMALL")

        out = tmp_path / "out"
        with patch('bagfd.client.download_files', return_value=[]):
            res = c.download("shared.bundle", platform="japan-android",
                             output_dir=out, filter_method="contains")
        assert res.count == 1                                    # no duplicate path
        assert (out / "shared.bundle").read_bytes() == b"SMALL"  # smaller pack won


# ---------------------------------------------------------------------------
# get_latest_files() surfaces an unavailable server as retryable
# ---------------------------------------------------------------------------

class TestResourceUnavailable:
    @pytest.fixture
    def client(self, tmp_path):
        c = BlueArchiveGameFilesDownloader(data_dir=tmp_path / "data")
        _seed_global(c.db_path)
        _seed_japan(c.db_path, "japan-android")
        _seed_versions(c.db_path)
        return c

    def test_global_short_delivery_raises(self, client, tmp_path):
        # download_files swallows failures and returns fewer paths than asked;
        # get_latest_files should surface that as ResourceUnavailableError.
        with patch('bagfd.client.download_files', return_value=[]):
            with pytest.raises(ResourceUnavailableError):
                client.get_latest_files('ch0230', platform='global-android',
                                        cache_dir=tmp_path / 'c')

    def test_japan_missing_pack_raises(self, client, tmp_path):
        # patched downloader places no zip -> the pack is missing -> retryable.
        with patch('bagfd.client.download_files', return_value=[]):
            with pytest.raises(ResourceUnavailableError):
                client.get_latest_files('ch0230_a.bundle', platform='japan-android',
                                        cache_dir=tmp_path / 'c', filter_method='contains')

    def test_japan_full_delivery_no_raise(self, client, tmp_path):
        # the pack is present -> extracts normally, no error.
        zpath = client.zip_cache / 'japan-android' / 'Pack_ch0230.zip'
        zpath.parent.mkdir(parents=True)
        with zipfile.ZipFile(zpath, 'w') as zf:
            zf.writestr('ch0230_a.bundle', b'AAA')
            zf.writestr('ch0230_b.bundle', b'BBB')
        with patch('bagfd.client.download_files', return_value=[]):
            paths = client.get_latest_files('ch0230_a.bundle', platform='japan-android',
                                            cache_dir=tmp_path / 'c', filter_method='contains')
        assert len(paths) == 1
