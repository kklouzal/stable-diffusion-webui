from __future__ import annotations
import os
import re

from modules import persistent_artifact_cache, shared
from modules.paths_internal import cwd


def natural_sort_key(s, regex=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower() for text in regex.split(s)]


def listfiles(dirname):
    filenames = [os.path.join(dirname, x) for x in sorted(os.listdir(dirname), key=natural_sort_key) if not x.startswith(".")]
    return [file for file in filenames if os.path.isfile(file)]


def walk_files(path, allowed_extensions=None):
    if not os.path.exists(path):
        return

    if allowed_extensions is not None:
        allowed_extensions = set(allowed_extensions)

    items = list(os.walk(path, followlinks=True))
    items = sorted(items, key=lambda x: natural_sort_key(x[0]))

    for root, _, files in items:
        for filename in sorted(files, key=natural_sort_key):
            if allowed_extensions is not None:
                _, ext = os.path.splitext(filename)
                if ext.lower() not in allowed_extensions:
                    continue

            if not shared.opts.list_hidden_files and ("/." in root or "\\." in root):
                continue

            yield os.path.join(root, filename)


def ldm_print(*args, **kwargs):
    if shared.opts.hide_ldm_prints:
        return

    print(*args, **kwargs)


def path_is_parent(parent_path, child_path):
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)
    try:
        return os.path.commonpath([parent_path, child_path]) == parent_path
    except ValueError:
        return False


def truncate_path(target_path, base_path=cwd):
    abs_target, abs_base = os.path.abspath(target_path), os.path.abspath(base_path)
    if path_is_parent(abs_base, abs_target):
        return os.path.relpath(abs_target, abs_base)
    return abs_target


def topological_sort(dependencies):
    """Accepts a dictionary mapping name to its dependencies, returns a list of names ordered according to dependencies.
    Ignores errors relating to missing dependencies or circular dependencies
    """

    visited = {}
    result = []

    def inner(name):
        visited[name] = True

        for dep in dependencies.get(name, []):
            if dep in dependencies and dep not in visited:
                inner(dep)

        result.append(name)

    for depname in dependencies:
        if depname not in visited:
            inner(depname)

    return result


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
    hash_prefix: str | None = None,
    re_download: bool = False,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.
    Returns the path to the downloaded file.

    file_name: if specified, it will be used as the filename, otherwise the filename will be extracted from the url.
        file is downloaded to {file_name}.tmp then moved to the final location after download is complete.
    hash_prefix: sha256 hex string, if provided, the hash of the downloaded file will be checked against this prefix.
        if the hash does not match, the temporary file is deleted and a ValueError is raised.
    re_download: forcibly re-download the file even if it already exists.
    """
    from urllib.parse import urlparse
    import requests
    from tqdm import tqdm

    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)

    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    if re_download or not os.path.exists(cached_file):
        os.makedirs(model_dir, exist_ok=True)
        temp_file = os.path.join(model_dir, f"{file_name}.tmp")
        print(f'\nDownloading: "{url}" to {cached_file}')
        # (connect, read) seconds: the read timeout bounds each wait for the next chunk, not the whole download, so a
        # stalled connection fails the request (which may hold the generation queue lock) instead of hanging it.
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name, disable=not progress) as progress_bar:
                with open(temp_file, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            file.write(chunk)
                            progress_bar.update(len(chunk))

        if hash_prefix and not compare_sha256(temp_file, hash_prefix):
            print(f"Hash mismatch for {temp_file}. Deleting the temporary file.")
            os.remove(temp_file)
            raise ValueError(f"File hash does not match the expected hash prefix {hash_prefix}!")

        os.rename(temp_file, cached_file)
    return cached_file


def compare_sha256(file_path: str, hash_prefix: str) -> bool:
    """Check if the SHA256 hash of the file matches the given prefix."""
    return persistent_artifact_cache.sha256_file(file_path).startswith(hash_prefix.strip().lower())
