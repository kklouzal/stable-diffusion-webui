from pathlib import Path


def test_controlnet_teed_runtime_patcher_removes_deprecated_torch_jit_script_decorators():
    patcher = Path('gb10/patch-controlnet-teed.py').read_text(encoding='utf8')

    assert "patched = original.replace('@torch.jit.script\\n', '')" in patcher
    assert "('Fsmish.py', 'Fmish.py')" in patcher
    assert 'deprecated torch.jit.script decorators' in patcher


def test_run_sh_applies_controlnet_teed_patch_to_mounted_production_extension():
    run_sh = Path('gb10/run.sh').read_text(encoding='utf8')

    assert 'CONTROLNET_ROOT="${HOST_ROOT}/Extensions/sd-webui-controlnet"' in run_sh
    assert 'patch-controlnet-teed.py' in run_sh
    assert 'annotator/teed' in run_sh
