"""refresh_vae_list(): natural name order and a deterministic winner for duplicate basenames."""

import os

import pytest

from test.helpers import init_shared

shared = init_shared()

from modules import sd_vae  # noqa: E402


@pytest.fixture
def vae_files(monkeypatch):
    """Serve fixed glob results in a scandir-like (unsorted) order."""
    root = os.path.join(os.sep, "vae")
    listing = {
        "first": [os.path.join(root, "x", "b.safetensors"), os.path.join(root, "a10.pt"), os.path.join(root, "z", "dup.pt"), os.path.join(root, "a2.pt"), os.path.join(root, "a", "dup.pt")],
        "second": [os.path.join(root, "late", "a2.pt")],
    }
    monkeypatch.setattr(sd_vae, "vae_search_paths", lambda: list(listing))
    monkeypatch.setattr(sd_vae.glob, "iglob", lambda pattern, recursive=False: iter(listing[pattern]))
    saved = dict(sd_vae.vae_dict)
    yield root
    sd_vae.vae_dict.clear()
    sd_vae.vae_dict.update(saved)


def test_refresh_vae_list_is_in_natural_name_order(vae_files):
    sd_vae.refresh_vae_list()

    assert list(sd_vae.vae_dict) == ["a2.pt", "a10.pt", "b.safetensors", "dup.pt"]


def test_refresh_vae_list_resolves_duplicates_deterministically(vae_files):
    sd_vae.refresh_vae_list()

    # later search paths still override earlier ones; within one path the lexicographically last file wins
    assert sd_vae.vae_dict["a2.pt"] == os.path.join(vae_files, "late", "a2.pt")
    assert sd_vae.vae_dict["dup.pt"] == os.path.join(vae_files, "z", "dup.pt")
