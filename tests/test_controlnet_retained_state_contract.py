from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "extensions/sd-webui-controlnet/scripts"


def _source(rel):
    return (SCRIPTS / rel).read_text(encoding="utf-8")


def test_source_covers_retained_state_owners():
    lllite = _source("controlnet_lllite.py")
    assert "_all_hack_lock = RLock()" in lllite
    assert 'getattr(k, "_controlnet_lllite_owner", None) is owned' in lllite
    ipadapter = _source("ipadapter/plugable_ipadapter.py")
    assert "_all_hacks_lock = RLock()" in ipadapter
    assert 'getattr(k, "_controlnet_ipadapter_owner", None) is owned' in ipadapter
    assert "def release_request_state(self):" in ipadapter
    hook = _source("hook.py")
    restore = hook[hook.index("    def restore(self):"):]
    assert "clear_all_lllite()" in restore
    assert "clear_all_ip_adapter()" in restore
    assert "release_request_state" in restore


def test_owner_token_prevents_stale_cleanup_from_clobbering_new_owner():
    class Module:
        pass
    module = Module()
    baseline = object()
    first_wrapper = object()
    second_wrapper = object()
    first_owner = {}
    second_owner = {}
    module.forward = first_wrapper
    module._controlnet_lllite_owner = first_owner
    first_owner[module] = baseline
    module.forward = second_wrapper
    module._controlnet_lllite_owner = second_owner
    second_owner[module] = first_wrapper

    owned = first_owner
    for target, original in owned.items():
        if getattr(target, "_controlnet_lllite_owner", None) is owned:
            target.forward = original
    assert module.forward is second_wrapper


def test_request_state_release_drops_derived_tensors_but_not_model_weights():
    class Adapter:
        def __init__(self):
            self.ipadapter = object()
            self.cache = {"k": object()}
            self.image_emb = object()
            self.effective_region_mask = object()
            self.latent_width = 64
            self.latent_height = 64
            self.pulid_attn_setting = object()
        def reset(self): self.cache = {}
        def release_request_state(self):
            self.reset(); self.image_emb = None; self.effective_region_mask = None
            self.latent_width = self.latent_height = 0; self.pulid_attn_setting = None
    adapter = Adapter(); weights = adapter.ipadapter
    adapter.release_request_state()
    assert adapter.ipadapter is weights
    assert adapter.cache == {} and adapter.image_emb is None
    assert adapter.effective_region_mask is None
    assert adapter.latent_width == adapter.latent_height == 0
    assert adapter.pulid_attn_setting is None
