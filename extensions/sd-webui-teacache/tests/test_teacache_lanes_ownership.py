"""TeaCache lane invalidation and UNet forward ownership (scripts/teacache.py); fixtures are in conftest.py.

CPU-only. A tiny UNet stand-in follows sgm ``UNetModel.forward``'s call structure (first block blind to the
cross-attention context, deeper blocks conditioned on it), so cache decisions and restore behaviour run through
the real patched forward.
"""

from types import SimpleNamespace

import pytest
import torch


class FakeUNet:
    """sgm UNetModel.forward shape: emb from timesteps (+ y), input_blocks[0..1] ignore the context."""

    model_channels = 4
    num_classes = "sequential"

    def __init__(self):
        self.deep_calls = []  # (step, lane ordinal) of every deep-path evaluation by TeaCache's own forward path
        self.session = None
        self.time_embed = lambda t_emb: t_emb
        self.label_emb = lambda y: y
        self.input_blocks = [
            lambda h, emb, ctx: h * 1.0,
            lambda h, emb, ctx: h + emb[:, :, None, None] * 0.5,
            self._recorded_deep_block,
        ]
        self.middle_block = lambda h, emb, ctx: h * 2.0
        self.output_blocks = [lambda h, emb, ctx: h[:, :4] + 0.25 * h[:, 4:] for _ in range(3)]
        self.out = lambda h: h

    @staticmethod
    def _deep_block(h, emb, ctx):
        return h + ctx.mean(dim=(1, 2))[:, None, None, None]

    def _recorded_deep_block(self, h, emb, ctx):
        cache = self.session
        self.deep_calls.append((None, None) if cache is None else (cache.current_step, cache.lane[1]))
        return self._deep_block(h, emb, ctx)

    def reference(self, x, timesteps=None, context=None, y=None, **kwargs):
        """Uncached forward with the same graph (what TeaCache must reproduce on every refresh); never recorded."""
        emb = self.time_embed(timesteps[:, None].float().expand(-1, 4) * 0.01) + self.label_emb(y)
        hs = []
        h = x
        for block in (*self.input_blocks[:2], self._deep_block):
            h = block(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for block in self.output_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), emb, context)
        return self.out(h.to(x.dtype))


def _patch(teacache, unet):
    unet._openclaw_teacache_original_forward = unet.reference
    unet._teacache_patched = True
    unet.forward = teacache.patched_forward.__get__(unet)


def _inputs(batch=2):
    x = torch.linspace(-1.0, 1.0, batch * 4 * 2 * 2).reshape(batch, 4, 2, 2)
    timesteps = torch.full((batch,), 500.0)
    y = torch.linspace(0.0, 1.0, batch * 4).reshape(batch, 4)
    return x, timesteps, y


def test_context_change_refreshes_lane_instead_of_reusing_stale_residual(teacache):
    """Prompt editing switches the context mid-run; SDXL's first block cannot see it, the cache key must."""
    unet = FakeUNet()
    _patch(teacache, unet)
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    unet.session = session
    teacache._set_cache(session)
    x, timesteps, y = _inputs()
    first_prompt = torch.zeros(2, 3, 8)
    second_prompt = torch.ones(2, 3, 8)

    unet.forward(x, timesteps=timesteps, context=first_prompt, y=y)
    teacache.next_step()
    switched = unet.forward(x, timesteps=timesteps, context=second_prompt, y=y)
    teacache.next_step()
    unchanged = unet.forward(x, timesteps=timesteps, context=second_prompt, y=y)

    # Refresh on the switch step: output is the true forward for the new prompt, not the old prompt's residual.
    assert torch.equal(switched, unet.reference(x, timesteps=timesteps, context=second_prompt, y=y))
    assert not torch.allclose(switched, unet.reference(x, timesteps=timesteps, context=first_prompt, y=y))
    # Same conditioning next step: the cache is used again (no deep evaluation at step 3).
    assert unet.deep_calls == [(1, 0), (2, 0)]
    torch.testing.assert_close(unchanged, unet.reference(x, timesteps=timesteps, context=second_prompt, y=y))


def test_vector_conditioning_change_refreshes_lane(teacache):
    unet = FakeUNet()
    _patch(teacache, unet)
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    unet.session = session
    teacache._set_cache(session)
    x, timesteps, y = _inputs()
    context = torch.zeros(2, 3, 8)

    unet.forward(x, timesteps=timesteps, context=context, y=y)
    teacache.next_step()
    # Swapped vector conditioning moves the first-block distance far less than the threshold, yet the cached
    # residual belongs to different conditioning.
    out = unet.forward(x, timesteps=timesteps, context=context, y=y.flip(0))

    assert unet.deep_calls == [(1, 0), (2, 0)]
    assert torch.equal(out, unet.reference(x, timesteps=timesteps, context=context, y=y.flip(0)))


def test_every_lane_is_bounded_by_max_consecutive_and_lanes_stay_coherent(teacache):
    """PAG's hidden pass replays the main call's inputs as its own lane: both lanes must refresh together."""
    unet = FakeUNet()
    _patch(teacache, unet)
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=2, start=0.0, end=1.0, steps=10)
    unet.session = session
    teacache._set_cache(session)
    x, timesteps, y = _inputs()
    context = torch.zeros(2, 3, 8)

    for _ in range(7):
        unet.forward(x, timesteps=timesteps, context=context, y=y)  # lane 0: main pass
        unet.forward(x, timesteps=timesteps, context=context, y=y)  # lane 1: PAG-style replay
        teacache.next_step()

    refreshed = {lane: [step for step, call_lane in unet.deep_calls if call_lane == lane] for lane in (0, 1)}
    # Refresh, two hits, refresh, ... for each lane. With the former shared lane-0 counter, lane 1 checked a
    # counter lane 0 had already advanced or reset in the same step and refreshed out of phase (steps 1, 3, 6):
    # on those steps one lane was fresh and the other stale, and a lane-0 distance refresh left lane 1 unbounded.
    assert refreshed[0] == [1, 4, 7]
    assert refreshed[1] == [1, 4, 7]


def test_alternating_ngms_batch_layouts_each_keep_their_own_lane(teacache):
    """NGMS drops the uncond on every other step: the cond+uncond batch and the cond-only batch alternate.

    Keyed by call index, call 0 alternated between the two signatures and never hit; keyed by signature, each
    layout reuses the residual of the last step that ran it.
    """
    unet = FakeUNet()
    _patch(teacache, unet)
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10)
    unet.session = session
    teacache._set_cache(session)
    x, timesteps, y = _inputs()
    context = torch.zeros(2, 3, 8)
    layouts = [(x, timesteps, context, y), (x[:1], timesteps[:1], context[:1], y[:1])]

    outputs = []
    for step in range(6):
        x_in, t_in, c_in, y_in = layouts[step % 2]
        outputs.append((unet.forward(x_in, timesteps=t_in, context=c_in, y=y_in), layouts[step % 2]))
        teacache.next_step()

    # Only the first step of each layout evaluates the deep path; every later step is a cache hit.
    assert unet.deep_calls == [(1, 0), (2, 0)]
    for out, (x_in, t_in, c_in, y_in) in outputs:
        torch.testing.assert_close(out, unet.reference(x_in, timesteps=t_in, context=c_in, y=y_in))
    assert sorted(lane[0][0][0] for lane in session.residuals) == [(1, 4, 2, 2), (2, 4, 2, 2)]


def test_session_counts_hits_per_lane(teacache):
    session = teacache.TeaCacheSession(threshold=1.0, max_consecutive=1, start=0.0, end=1.0, steps=10)
    signature = ((1, 2), "torch.float32", "cpu")
    for _ in range(2):
        session.update_condition(torch.ones(1, 2), signature)
        session.store_current_residual(torch.zeros(1, 2))
    session.next_step()
    decisions = []
    for _ in range(2):
        session.update_condition(torch.ones(1, 2), signature)
        decisions.append(session.use_cache)
    assert decisions == [True, True]
    assert session.consecutive_hits == {(signature, 0): 1, (signature, 1): 1}


class FakeControlNetHook:
    """The owner-scoped wrapper protocol of extensions/sd-webui-controlnet/scripts/hook.py (hook/restore)."""

    def __init__(self):
        self.token = object()
        self.baseline = None
        self.wrapper = None

    def hook(self, unet):
        assert getattr(unet, "_controlnet_forward_hook_owner", None) is None
        self.baseline = unet.forward
        self.wrapper = lambda *args, **kwargs: self.baseline(*args, **kwargs)
        unet._controlnet_forward_hook_owner = self.token
        unet.forward = self.wrapper

    def restore(self, unet):
        if getattr(unet, "_controlnet_forward_hook_owner", None) is self.token and unet.forward is self.wrapper:
            unet.forward = self.baseline
            del unet._controlnet_forward_hook_owner


def _processing(unet):
    return SimpleNamespace(
        sd_model=SimpleNamespace(model=SimpleNamespace(diffusion_model=unet), is_sdxl=True),
        extra_generation_params={},
        steps=20,
    )


@pytest.mark.parametrize("teacache_enabled_next", [True, False])
def test_stale_patch_under_controlnet_is_not_restored_across_its_wrapper(teacache, teacache_enabled_next):
    unet = FakeUNet()
    original = unet.reference
    unet.forward = original
    p = _processing(unet)
    script = teacache.TeaCacheScript()
    script.process(p, True)
    assert unet._teacache_patched and unet.forward is not original
    # The generation fails after process(): postprocess never runs and the patch stays installed.

    controlnet = FakeControlNetHook()
    controlnet.hook(unet)  # next request: ControlNet's process() runs before TeaCache's
    script.process(p, teacache_enabled_next)

    assert unet.forward is controlnet.wrapper
    assert unet._controlnet_forward_hook_owner is controlnet.token
    assert teacache._has_external_unet_forward_hook(p)

    script.process_before_every_sampling(p, teacache_enabled_next)
    x, timesteps, y = _inputs()
    context = torch.zeros(2, 3, 8)
    # The patch below ControlNet only delegates.
    assert torch.equal(unet.forward(x, timesteps=timesteps, context=context, y=y), original(x, timesteps=timesteps, context=context, y=y))
    assert unet.deep_calls == []

    script.postprocess(p)
    assert unet.forward is controlnet.wrapper

    controlnet.restore(unet)  # ControlNet's next process() restores its baseline: TeaCache's patch is on top again
    assert not hasattr(unet, "_controlnet_forward_hook_owner")
    script.process(p, False)

    assert unet.forward is original
    assert not unet._teacache_patched
    assert not hasattr(unet, "_openclaw_teacache_original_forward")
    assert teacache._get_cache() is None


def test_patched_forward_failure_under_wrapper_keeps_the_wrapper(teacache):
    unet = FakeUNet()

    def failing(*args, **kwargs):
        raise RuntimeError("synthetic model failure")

    unet._openclaw_teacache_original_forward = failing
    unet._teacache_patched = True
    unet.forward = teacache.patched_forward.__get__(unet)
    controlnet = FakeControlNetHook()
    controlnet.hook(unet)
    teacache._set_cache(teacache.TeaCacheSession(threshold=1.0, max_consecutive=0, start=0.0, end=1.0, steps=10))
    x, timesteps, y = _inputs()

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        unet.forward(x, timesteps=timesteps, context=torch.zeros(2, 3, 8), y=y)

    assert unet.forward is controlnet.wrapper
    assert unet._teacache_patched
    assert teacache._get_cache() is None


def test_patch_from_an_earlier_script_load_is_still_restored(teacache, load_teacache):
    unet = FakeUNet()
    original = unet.reference
    unet.forward = original
    p = _processing(unet)
    teacache.TeaCacheScript().process(p, True)

    reloaded = load_teacache("teacache_reloaded")
    reloaded.TeaCacheScript().process(p, False)

    assert unet.forward is original
    assert not unet._teacache_patched
