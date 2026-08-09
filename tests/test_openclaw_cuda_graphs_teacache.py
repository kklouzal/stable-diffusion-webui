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
