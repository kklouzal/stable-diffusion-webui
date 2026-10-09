"""ControlLLLite and IP-Adapter hack cleanup restores only the blocks its owner dict still owns, under the module lock;
IP-Adapter request-state release keeps the weights. UnetHook.restore() calling these is in test_controlnet_hook_lifecycle.py."""
import ast
import importlib.util
import threading
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "extensions/sd-webui-controlnet/scripts"


def _load_lllite():
    # controlnet_lllite.py imports only re, threading and torch.
    spec = importlib.util.spec_from_file_location("controlnet_lllite_under_test", SCRIPTS / "controlnet_lllite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_ipadapter_hacks():
    """The real hack registry of plugable_ipadapter.py (its package-relative imports keep it from loading alone)."""
    path = SCRIPTS / "ipadapter/plugable_ipadapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    registry = [node for node in tree.body if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in {"all_hacks", "current_model", "_all_hacks_lock"} for target in node.targets)]
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"hack_blk", "clear_all_ip_adapter"}]
    adapter = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PlugableIPAdapter")
    methods = [node for node in adapter.body if isinstance(node, ast.FunctionDef) and node.name in {"reset", "release_request_state"}]
    assert len(registry) == 3 and len(functions) == 2 and len(methods) == 2
    module = ast.Module(body=[*registry, *functions, ast.ClassDef(name="PlugableIPAdapter", bases=[], keywords=[], body=methods, decorator_list=[])],
                        type_ignores=[])
    ast.fix_missing_locations(module)

    def attn_forward_hacked(self, x, context=None, **kwargs):
        return "hacked"

    namespace = {"RLock": threading.RLock, "attn_forward_hacked": attn_forward_hacked}
    exec(compile(module, str(path), "exec"), namespace)
    return types.SimpleNamespace(namespace=namespace, **namespace)


def _assert_clear_waits_for_lock(lock, clear):
    lock.acquire()
    try:
        worker = threading.Thread(target=clear)
        worker.start()
        worker.join(0.05)
        assert worker.is_alive()
    finally:
        lock.release()
    worker.join(2)
    assert not worker.is_alive()


def _lllite_unet(lllite, dim=16):
    module = lllite.LLLiteModule("m", is_conv2d=False, in_dim=dim, depth=1, cond_emb_dim=8, mlp_dim=12)
    state = {f"lllite_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q.{k}": v for k, v in module.state_dict().items()}
    to_q = torch.nn.Linear(dim, dim, bias=False)
    block = types.SimpleNamespace(transformer_blocks=[types.SimpleNamespace(attn1=types.SimpleNamespace(to_q=to_q))])
    unet = types.SimpleNamespace(input_blocks=[None, None, None, None, [None, block]], current_sampling_percent=0.5,
                                 is_in_high_res_fix=False, current_h_shape=(1, dim, 8, 8))
    return state, unet, to_q


def test_lllite_cleanup_restores_only_the_blocks_its_owner_still_owns():
    lllite = _load_lllite()
    state, unet, to_q = _lllite_unet(lllite)
    hint = torch.rand(1, 3, 64, 64)
    baseline = to_q.forward

    first, second = lllite.PlugableControlLLLite(state), lllite.PlugableControlLLLite(state)
    first.hook(model=unet, cond=hint, weight=0.6, start=0.0, end=1.0)
    owner, wrapper = lllite.all_hack, to_q.forward
    second.hook(model=unet, cond=hint, weight=0.4, start=0.0, end=1.0)
    assert to_q.forward is wrapper and len(to_q.lllite_list) == 2
    assert to_q._controlnet_lllite_owner is owner and owner == {to_q: baseline}

    _assert_clear_waits_for_lock(lllite._all_hack_lock, lllite.clear_all_lllite)
    assert to_q.forward == baseline and to_q.lllite_list == []
    assert not hasattr(to_q, "_controlnet_lllite_owner")
    assert lllite.all_hack == {} and lllite.all_hack is not owner

    # A block another owner took over since is left to that owner.
    first.hook(model=unet, cond=hint, weight=0.6, start=0.0, end=1.0)
    newer_owner, newer_forward = {}, object()
    to_q._controlnet_lllite_owner, to_q.forward = newer_owner, newer_forward
    lllite.clear_all_lllite()
    assert to_q.forward is newer_forward and to_q._controlnet_lllite_owner is newer_owner


def test_ipadapter_cleanup_restores_only_the_blocks_its_owner_still_owns():
    ipadapter = _load_ipadapter_hacks()
    registry = ipadapter.namespace

    class Attn:
        def forward(self, x, context=None):
            return "baseline"

    block, taken = Attn(), Attn()
    for target in (block, taken):
        ipadapter.hack_blk(target, "patch-1", Attn)
    ipadapter.hack_blk(block, "patch-2", Attn)
    owner = registry["all_hacks"]
    assert block.forward(None) == "hacked" and block.ipadapter_hacks == ["patch-1", "patch-2"]
    assert block._controlnet_ipadapter_owner is owner and owner[block] == Attn.forward.__get__(block)

    newer_owner, newer_forward = {}, object()
    taken._controlnet_ipadapter_owner, taken.forward = newer_owner, newer_forward
    registry["current_model"] = object()
    _assert_clear_waits_for_lock(registry["_all_hacks_lock"], registry["clear_all_ip_adapter"])

    assert block.forward(None) == "baseline" and block.ipadapter_hacks == []
    assert not hasattr(block, "_controlnet_ipadapter_owner")
    assert taken.forward is newer_forward and taken._controlnet_ipadapter_owner is newer_owner
    assert registry["all_hacks"] == {} and registry["all_hacks"] is not owner
    assert registry["current_model"] is None


def test_ipadapter_request_state_release_drops_derived_state_but_not_model_weights():
    adapter = _load_ipadapter_hacks().PlugableIPAdapter()
    weights = object()
    adapter.__dict__.update(ipadapter=weights, cache={"k": object()}, region_masks={"r": object()}, cond_rows_memo=object(),
                            image_emb=object(), effective_region_mask=object(), latent_width=64, latent_height=48)
    adapter.release_request_state()
    assert adapter.ipadapter is weights
    assert adapter.cache == {} and adapter.region_masks == {} and adapter.cond_rows_memo is None
    assert adapter.image_emb is None and adapter.effective_region_mask is None
    assert adapter.latent_width == adapter.latent_height == 0
