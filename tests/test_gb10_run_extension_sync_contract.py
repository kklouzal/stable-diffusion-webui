import re
from pathlib import Path


def test_owned_extension_sync_protects_runtime_data_contents():
    # gb10/run.sh mirrors each owned extension with rsync --delete. The checkout keeps its own (git-ignored,
    # empty) openclaw-multi-sampler/data/ and sd-webui-controlnet/annotator/downloads/ directories, and a plain
    # 'P /data/' rule protects only the directory entry: rsync then descends and deletes the saved chains and
    # downloaded weights. Verified with rsync 3.2.7: only the /*** form keeps the files in both layouts.
    # ControlNet models/ is protected the same way, so a model placed only on the host survives a deploy.
    source = Path("gb10/run.sh").read_text()
    rules = re.findall(r"--filter '(P [^']+)'", source)
    assert "P /data/***" in rules
    assert "P /annotator/downloads/***" in rules
    assert "P /models/***" in rules
    assert not {"P /data/", "P /annotator/downloads/", "P /models/"} & set(rules)


def test_owned_extension_sync_does_not_deploy_tool_caches():
    source = Path("gb10/run.sh").read_text()
    excludes = re.findall(r"--exclude '([^']+)'", source[source.index("rsync -a --checksum --delete"):])
    for cache in (".git/", "__pycache__/", "*.pyc", ".ruff_cache/", ".pytest_cache/"):
        assert cache in excludes
