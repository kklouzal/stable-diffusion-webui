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


def test_epoch_transaction_blocks_generic_bump_until_cache_publication():
    import threading

    import modules.openclaw_cache_epochs as epochs

    epochs.epoch_registry.reset()
    compute_entered = threading.Event()
    allow_publish = threading.Event()
    bump_started = threading.Event()
    bump_finished = threading.Event()
    cache = [None, None]
    captured = []

    def cache_transaction():
        with epochs.epoch_transaction():
            snapshot = epochs.epoch_subset(('conditioner_epoch',))
            compute_entered.set()
            assert allow_publish.wait(2)
            cache[:] = [snapshot, 'computed']
            captured.append(epochs.epoch_subset(('conditioner_epoch',)))

    def mutate_dependency():
        assert compute_entered.wait(2)
        bump_started.set()
        epochs.bump_epoch('conditioner_epoch', reason='conditioning_cleared')
        bump_finished.set()

    cache_thread = threading.Thread(target=cache_transaction)
    bump_thread = threading.Thread(target=mutate_dependency)
    cache_thread.start()
    bump_thread.start()
    assert bump_started.wait(2)
    assert not bump_finished.wait(0.05)
    allow_publish.set()
    cache_thread.join(2)
    bump_thread.join(2)

    assert not cache_thread.is_alive() and not bump_thread.is_alive()
    assert cache[0] == captured[0] == (('conditioner_epoch', 0),)
    assert epochs.epoch_subset(('conditioner_epoch',)) == (('conditioner_epoch', 1),)


def test_ti_and_clear_commits_use_epoch_first_lock_order():
    epochs_src = text('modules/openclaw_cache_epochs.py')
    assert 'def epoch_transaction()' in epochs_src
    assert 'self._lock = threading.RLock()' in epochs_src

    processing = text('modules/processing.py')
    cache_block = processing[processing.index('def get_conds_with_caching'):processing.index('def setup_conds')]
    transaction = cache_block.index('with openclaw_cache_epochs.epoch_transaction():')
    cache_lock = cache_block.index('with StableDiffusionProcessing.conditioning_cache_lock:')
    capture = cache_block.index('cached_params = self.cached_params')
    publish = cache_block.index('cache[:] = [cached_params, computed]')
    assert transaction < capture < cache_lock < publish

    ti = text('modules/textual_inversion/textual_inversion.py')
    ti_block = ti[ti.index('def load_textual_inversion_embeddings'):ti.index('def find_embedding_at_position')]
    transaction = ti_block.index('with openclaw_cache_epochs.epoch_transaction():')
    publication = ti_block.index('with self._publication_lock:')
    maps = ti_block.index('self.ids_lookup, self.word_embeddings, self.skipped_embeddings')
    ti_epoch = ti_block.index('bump_epoch("textual_inversion_epoch"')
    tokenizer_epoch = ti_block.index('bump_epoch("tokenizer_epoch"')
    assert transaction < publication < maps < ti_epoch < tokenizer_epoch

    clear = text('extensions/openclaw-clear-cond-cache/scripts/openclaw_clear_cond_cache.py')
    clear_block = clear[clear.index('def clear_cond_cache'):clear.index('def on_app_started')]
    transaction = clear_block.index('with openclaw_cache_epochs.epoch_transaction():')
    cache_lock = clear_block.index('with StableDiffusionProcessing.conditioning_cache_lock:')
    slot_clear = clear_block.index('StableDiffusionProcessing.cached_c = [None, None]')
    conditioner_epoch = clear_block.index('bump_epoch("conditioner_epoch"')
    hook_epoch = clear_block.index('bump_epoch("conditioning_hook_epoch"')
    assert transaction < cache_lock < slot_clear < conditioner_epoch < hook_epoch


def test_epoch_transaction_exposes_only_coherent_ti_and_clear_states():
    import threading

    import modules.openclaw_cache_epochs as epochs

    epochs.epoch_registry.reset()
    state = {'maps': 'old', 'cache': 'populated'}
    ti_maps_published = threading.Event()
    allow_ti_epochs = threading.Event()
    clear_cache_published = threading.Event()
    allow_clear_epochs = threading.Event()
    observations = []

    def observe(label):
        with epochs.epoch_transaction():
            observations.append((label, state.copy(), dict(epochs.epoch_subset((
                'textual_inversion_epoch', 'tokenizer_epoch',
                'conditioner_epoch', 'conditioning_hook_epoch',
            )))))

    def publish_ti():
        with epochs.epoch_transaction():
            state['maps'] = 'new'
            ti_maps_published.set()
            assert allow_ti_epochs.wait(2)
            epochs.bump_epoch('textual_inversion_epoch', reason='textual_inversion_reloaded')
            epochs.bump_epoch('tokenizer_epoch', reason='textual_inversion_reloaded')

    ti_thread = threading.Thread(target=publish_ti)
    ti_thread.start()
    assert ti_maps_published.wait(2)
    ti_observer = threading.Thread(target=observe, args=('ti',))
    ti_observer.start()
    ti_observer.join(0.05)
    assert ti_observer.is_alive()
    allow_ti_epochs.set()
    ti_thread.join(2)
    ti_observer.join(2)

    def clear_conditioning():
        with epochs.epoch_transaction():
            state['cache'] = 'empty'
            clear_cache_published.set()
            assert allow_clear_epochs.wait(2)
            epochs.bump_epoch('conditioner_epoch', reason='conditioning_cleared')
            epochs.bump_epoch('conditioning_hook_epoch', reason='conditioning_hook_changed')

    clear_thread = threading.Thread(target=clear_conditioning)
    clear_thread.start()
    assert clear_cache_published.wait(2)
    clear_observer = threading.Thread(target=observe, args=('clear',))
    clear_observer.start()
    clear_observer.join(0.05)
    assert clear_observer.is_alive()
    allow_clear_epochs.set()
    clear_thread.join(2)
    clear_observer.join(2)

    by_label = {label: (values, snapshot) for label, values, snapshot in observations}
    assert by_label['ti'][0]['maps'] == 'new'
    assert by_label['ti'][1]['textual_inversion_epoch'] == 1
    assert by_label['ti'][1]['tokenizer_epoch'] == 1
    assert by_label['clear'][0]['cache'] == 'empty'
    assert by_label['clear'][1]['conditioner_epoch'] == 1
    assert by_label['clear'][1]['conditioning_hook_epoch'] == 1
