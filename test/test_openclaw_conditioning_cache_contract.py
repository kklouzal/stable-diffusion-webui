import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSING = ROOT / "modules" / "processing.py"


def _cached_params_return_elements():
    tree = ast.parse(PROCESSING.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StableDiffusionProcessing")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "cached_params")
    ret = next(node for node in ast.walk(fn) if isinstance(node, ast.Return))
    return {ast.unparse(item) for item in ret.value.elts}


def test_conditioning_key_covers_parser_tokenization_and_effective_network_state():
    elements = _cached_params_return_elements()
    required = {
        "cache_namespace",
        "effective_network_state",
        "opts.CLIP_stop_at_last_layers",
        "opts.sdxl_clip_l_skip",
        "opts.emphasis",
        "opts.use_old_emphasis_implementation",
        "opts.comma_padding_backtrack",
        "required_prompts",
        "extra_network_data",
    }
    assert required <= elements


def test_conditioning_key_does_not_use_lora_transition_epoch():
    source = PROCESSING.read_text()
    start = source.index("    def cached_params(")
    end = source.index("    def get_conds_with_caching(", start)
    assert '"lora_applied_epoch"' not in source[start:end]
    assert "effective_network_state = self.active_lora_cond_signature()" in source[start:end]
    assert "current_network_state_identity" in source


def test_c_uc_namespaces_and_bounded_atomic_cache_contract_are_explicit():
    source = PROCESSING.read_text()
    assert 'get_conds_with_caching("uc"' in source
    assert 'get_conds_with_caching("c"' in source
    assert "conditioning_cache_lock = threading.RLock()" in source
    assert "with openclaw_cache_epochs.epoch_transaction():" in source
    assert "capacity=4" in source
    assert "cache[:] = [cached_params, computed]" in source


def test_miss_telemetry_reports_dependency_class_without_hot_path_tensor_work():
    source = PROCESSING.read_text()
    assert "_conditioning_cache_miss_reason" in source
    assert '"network_state"' in source
    assert '"comma_padding_backtrack"' in source
    assert 'reason = self._conditioning_cache_miss_reason(cache[0], cached_params)' in source
