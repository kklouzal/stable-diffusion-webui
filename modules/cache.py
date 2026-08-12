import json
import hashlib
import marshal
import os
import os.path
import threading
from contextlib import contextmanager

import diskcache
import tqdm

from modules.paths import data_path, script_path

cache_filename = os.environ.get('SD_WEBUI_CACHE_FILE', os.path.join(data_path, "cache.json"))
cache_dir = os.environ.get('SD_WEBUI_CACHE_DIR', os.path.join(data_path, "cache"))
caches = {}
cache_lock = threading.Lock()
_entry_locks = {}
_entry_locks_lock = threading.Lock()


def file_revision(stat_result):
    return {"device": stat_result.st_dev, "inode": stat_result.st_ino, "size": stat_result.st_size, "mtime_ns": stat_result.st_mtime_ns, "ctime_ns": stat_result.st_ctime_ns}


def callable_revision(func):
    code = getattr(func, "__code__", None)
    if code is None:
        return f"{getattr(func, '__module__', '')}:{getattr(func, '__qualname__', repr(func))}"
    return hashlib.sha256(marshal.dumps(code)).hexdigest()


@contextmanager
def entry_lock(subsection, title):
    key = (subsection, title)
    with _entry_locks_lock:
        lock = _entry_locks.setdefault(key, threading.RLock())
    with lock:
        yield


def dump_cache():
    """old function for dumping cache to disk; does nothing since diskcache."""

    pass


def make_cache(subsection: str) -> diskcache.Cache:
    return diskcache.Cache(
        os.path.join(cache_dir, subsection),
        size_limit=2**32,  # 4 GB, culling oldest first
        disk_min_file_size=2**18,  # keep up to 256KB in Sqlite
    )


def convert_old_cached_data():
    try:
        with open(cache_filename, "r", encoding="utf8") as file:
            data = json.load(file)
    except FileNotFoundError:
        return
    except Exception:
        os.replace(cache_filename, os.path.join(script_path, "tmp", "cache.json"))
        print('[ERROR] issue occurred while trying to read cache.json; old cache has been moved to tmp/cache.json')
        return

    total_count = sum(len(keyvalues) for keyvalues in data.values())

    with tqdm.tqdm(total=total_count, desc="converting cache") as progress:
        for subsection, keyvalues in data.items():
            cache_obj = caches.get(subsection)
            if cache_obj is None:
                cache_obj = make_cache(subsection)
                caches[subsection] = cache_obj

            for key, value in keyvalues.items():
                cache_obj[key] = value
                progress.update(1)


def cache(subsection):
    """
    Retrieves or initializes a cache for a specific subsection.

    Parameters:
        subsection (str): The subsection identifier for the cache.

    Returns:
        diskcache.Cache: The cache data for the specified subsection.
    """

    cache_obj = caches.get(subsection)
    if cache_obj is None:
        with cache_lock:
            if not os.path.exists(cache_dir) and os.path.isfile(cache_filename):
                convert_old_cached_data()

            cache_obj = caches.get(subsection)
            if cache_obj is None:
                cache_obj = make_cache(subsection)
                caches[subsection] = cache_obj

    return cache_obj


def cached_data_for_file(subsection, title, filename, func, *, source_revision=None, schema_revision=None):
    """
    Retrieves or generates data for a specific file, using a caching mechanism.

    Parameters:
        subsection (str): The subsection of the cache to use.
        title (str): The title of the data entry in the subsection of the cache.
        filename (str): The path to the file to be checked for modifications.
        func (callable): A function that generates the data if it is not available in the cache.

    Returns:
        dict or None: The cached or generated data, or None if data generation fails.

    The `cached_data_for_file` function implements a caching mechanism for data stored in files.
    It checks if the data associated with the given `title` is present in the cache and compares the
    modification time of the file with the cached modification time. If the file has been modified,
    the cache is considered invalid and the data is regenerated using the provided `func`.
    Otherwise, the cached data is returned.

    If the data generation fails, None is returned to indicate the failure. Otherwise, the generated
    or cached data is returned as a dictionary.
    """

    existing_cache = cache(subsection)
    with entry_lock(subsection, title):
        before = os.stat(filename)
        revision = source_revision() if source_revision else file_revision(before)
        schema = schema_revision if schema_revision is not None else callable_revision(func)
        entry = existing_cache.get(title)
        if entry and entry.get("source_revision") == revision and entry.get("schema_revision") == schema and "value" in entry:
            return entry["value"]
        value = func()
        if value is None:
            return None
        after = os.stat(filename)
        after_revision = source_revision() if source_revision else file_revision(after)
        if revision != after_revision:
            return None
        existing_cache[title] = {"source_revision": revision, "schema_revision": schema, "value": value}
        dump_cache()
        return value
