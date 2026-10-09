import hashlib
import marshal
import os
import os.path
import threading
from contextlib import contextmanager

import diskcache

from modules.paths_internal import cache_dir

caches = {}
cache_lock = threading.Lock()
_entry_locks = {}
_entry_locks_lock = threading.Lock()


def file_revision(stat_result):
    return {"device": stat_result.st_dev, "inode": stat_result.st_ino, "size": stat_result.st_size, "mtime_ns": stat_result.st_mtime_ns, "ctime_ns": stat_result.st_ctime_ns}


def file_cache_key(path, *extra):
    """In-memory cache key that changes with the file's content: (abspath, revision, *extra).

    The revision is file_revision() as a tuple. Size and mtime alone keep the key of a file replaced by one of equal
    size with its mtime preserved (cp -p, rsync -t, tar x); the replacement still changes the inode or the ctime.
    An unreadable file gets None as its revision instead of raising.
    """
    filename = os.path.abspath(path)
    try:
        file_identity = tuple(file_revision(os.stat(filename)).values())
    except OSError:
        file_identity = None
    return (filename, file_identity, *extra)


def drop_stale_file_entries(entries, current_key):
    """Delete entries keyed by file_cache_key() for the same file as `current_key` but a different key."""
    for key in list(entries):
        if key != current_key and key[0] == current_key[0]:
            del entries[key]


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


def make_cache(subsection: str) -> diskcache.Cache:
    return diskcache.Cache(
        os.path.join(cache_dir, subsection),
        size_limit=2**32,  # 4 GB, culling oldest first
        disk_min_file_size=2**18,  # keep up to 256KB in Sqlite
    )


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
            cache_obj = caches.get(subsection)
            if cache_obj is None:
                cache_obj = make_cache(subsection)
                caches[subsection] = cache_obj

    return cache_obj


def cached_data_for_file(subsection, title, filename, func, *, source_revision=None, schema_revision=None):
    """
    Return func()'s value for `filename`, cached in `subsection` under `title`.

    The cached value is reused while both revisions recorded with it are unchanged: the file's revision
    (file_revision() of its stat, or `source_revision()` when given) and the schema revision (`schema_revision`
    when given, else the code hash of `func`, so editing `func` invalidates its entries). Otherwise func() runs under
    the entry's lock. Its value is stored and returned, except that None is returned without caching when func()
    returns None or the file's revision changed while it ran. Exceptions from func() propagate.
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
        return value
