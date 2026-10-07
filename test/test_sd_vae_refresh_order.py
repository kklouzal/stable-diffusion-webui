"""refresh_vae_list(): natural name order and a deterministic winner for duplicate basenames."""

import os
import sys

import pytest

from modules import shared, shared_init

if getattr(shared, "opts", None) is None:
    shared_init.initialize()

from modules import processing, sd_vae  # noqa: E402


@pytest.fixture(autouse=True)
def _real_webui_modules_package(monkeypatch):
    # Other test files leave stub "modules" packages in sys.modules.
    monkeypatch.setitem(sys.modules, "modules", processing.modules)


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
