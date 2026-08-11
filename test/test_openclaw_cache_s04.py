from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def text(path): return (ROOT / path).read_text()

def test_conditioning_key_contract_and_isolation():
    src=text('modules/processing.py')
    for epoch in ('checkpoint_object_epoch','conditioner_epoch','textual_inversion_epoch','tokenizer_epoch','conditioning_hook_epoch','lora_applied_epoch','precision_epoch','device_epoch'):
        assert f'"{epoch}"' in src
    for namespace in ('"c"','"uc"','"hr_c"','"hr_uc"'):
        assert f'get_conds_with_caching({namespace}' in src
    assert 'seed' not in src[src.index('def cached_params'):src.index('def active_lora_cond_signature')]
    assert 'sampler_name' not in src[src.index('def cached_params'):src.index('def active_lora_cond_signature')]
    assert 'cache[:] = [cached_params, computed]' in src
    assert src.index('computed = function') < src.index('cache[:] = [cached_params, computed]')

def test_conditioning_exception_cannot_publish():
    src=text('modules/processing.py')
    block=src[src.index('def get_conds_with_caching'):src.index('def setup_conds')]
    assert 'except Exception:' in block and 'raise' in block
    assert block.index('except Exception:') < block.index('cache[:] = [cached_params, computed]')

def test_token_memo_is_call_local_and_sanitized():
    src=text('modules/sd_hijack_clip.py')
    block=src[src.index('def process_texts'):src.index('def forward',src.index('def process_texts'))]
    assert 'cache = {}' in block
    assert 'self.cache' not in block and 'global cache' not in block
    assert 'observe("E06", "hit"' in block and 'observe("E06", "miss"' in block
    assert 'semantic_key=semantic_key' in block

def test_ti_atomic_publication_and_exact_epoch_bumps():
    src=text('modules/textual_inversion/textual_inversion.py')
    block=src[src.index('def load_textual_inversion_embeddings'):src.index('def find_embedding_at_position')]
    assert 'staged = EmbeddingDatabase()' in block
    assert 'if new_signature == old_signature:' in block and 'return False' in block
    publish=block.index('self.ids_lookup, self.word_embeddings, self.skipped_embeddings')
    assert publish < block.index('bump_epoch("textual_inversion_epoch"')
    assert publish < block.index('bump_epoch("tokenizer_epoch"')
    assert block.count('bump_epoch(') == 2

def test_clear_hook_bumps_after_committed_clear():
    src=text('extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py')
    block=src[src.index('def clear_cond_cache'):src.index('def on_app_started')]
    clear=block.index('StableDiffusionProcessing.cached_c = [None, None]')
    assert clear < block.index('bump_epoch("conditioner_epoch"')
    assert clear < block.index('bump_epoch("conditioning_hook_epoch"')

def test_s04_telemetry_is_opaque():
    for path,family in [('modules/processing.py','E05'),('modules/sd_hijack_clip.py','E06'),('modules/textual_inversion/textual_inversion.py','E07')]:
        src=text(path)
        assert f'observe("{family}"' in src
        assert 'semantic_key=' in src
    epochs=text('modules/openclaw_cache_epochs.py')
    assert '"E05", "E06", "E07"' in epochs

def test_each_conditioning_epoch_changes_atomic_subset(monkeypatch):
    import modules.openclaw_cache_epochs as epochs
    relevant = (
        'checkpoint_object_epoch', 'conditioner_epoch', 'textual_inversion_epoch',
        'tokenizer_epoch', 'conditioning_hook_epoch', 'lora_applied_epoch',
        'precision_epoch', 'device_epoch',
    )
    epochs.epoch_registry.reset()
    baseline = epochs.epoch_subset(relevant)
    for dimension in relevant:
        epochs.epoch_registry.reset()
        epochs.bump_epoch(dimension, reason='manual')
        assert epochs.epoch_subset(relevant) != baseline
    for dimension in ('vae_object_epoch', 'vae_bytes_epoch'):
        epochs.epoch_registry.reset()
        epochs.bump_epoch(dimension, reason='manual')
        assert epochs.epoch_subset(relevant) == baseline
