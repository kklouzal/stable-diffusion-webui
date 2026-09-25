from types import SimpleNamespace

from modules import openclaw_cuda_graphs


def _denoiser_with_unet(unet):
    return SimpleNamespace(
        mask=None,
        nmask=None,
        p=SimpleNamespace(
            mask=None,
            nmask=None,
            sd_model=SimpleNamespace(model=SimpleNamespace(diffusion_model=unet)),
        ),
    )


def test_cuda_graphs_bypass_active_teacache_patch():
    unet = SimpleNamespace(_teacache_patched=True)

    assert openclaw_cuda_graphs._graph_denoiser_bypass_reason(_denoiser_with_unet(unet)) == "teacache_unet_forward_hook"


def test_cuda_graphs_bypass_active_teacache_original_forward_attr():
    unet = SimpleNamespace(_openclaw_teacache_original_forward=object())

    assert openclaw_cuda_graphs._graph_denoiser_bypass_reason(_denoiser_with_unet(unet)) == "teacache_unet_forward_hook"


def test_cuda_graph_key_distinguishes_active_teacache_patch_state():
    disabled = openclaw_cuda_graphs._denoiser_graph_key(_denoiser_with_unet(SimpleNamespace()))
    active_patch = openclaw_cuda_graphs._denoiser_graph_key(_denoiser_with_unet(SimpleNamespace(_teacache_patched=True)))
    active_original = openclaw_cuda_graphs._denoiser_graph_key(_denoiser_with_unet(SimpleNamespace(_openclaw_teacache_original_forward=object())))

    assert disabled != active_patch
    assert disabled != active_original
    assert active_patch[-2:] == (True, False)
    assert active_original[-2:] == (False, True)



def test_controlnet_owner_marker_bypasses_cuda_graph_capture():
    owner = object()
    unet = SimpleNamespace(_controlnet_forward_hook_owner=owner)

    assert openclaw_cuda_graphs._graph_denoiser_bypass_reason(_denoiser_with_unet(unet)) == "external_unet_forward_hook"
    assert unet._controlnet_forward_hook_owner is owner


def test_cuda_graphs_bypass_seg_attention_hooks_even_when_opted_in(monkeypatch):
    monkeypatch.setattr(openclaw_cuda_graphs, "_allow_seg_graphs", lambda: True)
    seg_params = SimpleNamespace(
        seg_active=True,
        seg_blur_sigma=3.0,
        seg_blur_threshold=10.0,
        seg_start_step=0,
        seg_end_step=7,
    )
    denoiser = _denoiser_with_unet(SimpleNamespace())
    denoiser.total_steps = 8
    denoiser.p.incant_cfg_params = {"seg_params": seg_params}

    assert openclaw_cuda_graphs._graph_denoiser_bypass_reason(denoiser) == "seg_attention_hooks"
