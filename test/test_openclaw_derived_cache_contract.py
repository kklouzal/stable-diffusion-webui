import ast
import os
import threading
import time
import types

from test.helpers import ROOT, load_source, module


class MemoryCache(dict):
    def __init__(self, *args, **kwargs):
        super().__init__()


def load_cache(tmp_path):
    return load_source("modules.cache", "modules/cache.py", {
        "diskcache": module("diskcache", Cache=MemoryCache),
        "tqdm": module("tqdm", tqdm=lambda *a, **k: None),
        "modules": module("modules", package=True),
        "modules.paths": module("modules.paths", data_path=str(tmp_path), script_path=str(tmp_path)),
    })


def test_file_cache_detects_same_size_same_mtime_replacement_and_schema(tmp_path):
    cache = load_cache(tmp_path)
    source = tmp_path / "source"
    source.write_text("one")
    original_mtime = source.stat().st_mtime_ns
    calls = []

    def parse():
        calls.append(1)
        return source.read_text()

    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1") == "one"
    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1") == "one"
    replacement = tmp_path / "replacement"
    replacement.write_text("two")
    os.utime(replacement, ns=(original_mtime, original_mtime))
    os.replace(replacement, source)
    os.utime(source, ns=(original_mtime, original_mtime))
    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1") == "two"
    assert cache.cached_data_for_file("s", "t", source, parse, schema_revision="v2") == "two"
    assert len(calls) == 3


def test_file_cache_single_publication_under_concurrency(tmp_path):
    cache = load_cache(tmp_path)
    source = tmp_path / "source"
    source.write_text("data")
    calls = 0
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def parse():
        nonlocal calls
        with lock:
            calls += 1
        time.sleep(0.02)
        return source.read_text()

    results = []
    def worker():
        barrier.wait()
        results.append(cache.cached_data_for_file("s", "t", source, parse, schema_revision="v1"))
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert results == ["data"] * 8
    assert calls == 1


def test_git_revision_tracks_head_ref_content(tmp_path):
    shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace(), opts=types.SimpleNamespace())
    extension = load_source("derived_extensions", "modules/extensions.py", {
        # `from modules import shared, errors, cache, scripts` reads these package attributes.
        "modules": module("modules", package=True, shared=shared, errors=types.SimpleNamespace(report=lambda *a, **k: None), cache=object(), scripts=types.SimpleNamespace(ScriptFile=object)),
        "modules.gitpython_hack": module("modules.gitpython_hack", Repo=object),
        "modules.paths_internal": module("modules.paths_internal", extensions_dir=str(tmp_path / "exts"), extensions_builtin_dir=str(tmp_path / "builtin")),
    })
    repo = tmp_path / "repo"; gitdir = repo / ".git"; (gitdir / "refs/heads").mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n"); (gitdir / "refs/heads/main").write_text("a" * 40 + "\n")
    first = extension.git_repository_revision(repo)
    (gitdir / "refs/heads/main").write_text("b" * 40 + "\n")
    assert extension.git_repository_revision(repo) != first


def _definitions(path, namespace, *names, methods=None):
    """Exec the named top-level definitions (functions, classes, assignments) of a source file in namespace. A class
    named in methods keeps only those methods and drops its bases, so it runs without the module's imports."""
    methods = methods or {}
    body = []
    for node in ast.parse((ROOT / path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            name = getattr(node, "name", None)
        if name in methods:
            node.bases, node.keywords = [], []
            node.body = [item for item in node.body if getattr(item, "name", None) in methods[name]]
        elif name not in names:
            continue
        body.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(ROOT / path), "exec"), namespace)
    assert set(names) | set(methods) <= namespace.keys()
    return namespace


def test_sampler_resolver_cache_is_bounded_and_dropped_when_the_registry_is_rebuilt(initialize):
    from modules import sd_samplers

    resolver = sd_samplers._resolve_sampler_and_scheduler
    # Request strings are part of the key, so the cache must be bounded ...
    assert resolver.cache_info().maxsize == 256
    sd_samplers.get_sampler_and_scheduler("Euler a", None)
    assert resolver.cache_info().currsize
    # ... and a rebuilt registry drops the entries resolved against the previous one.
    sd_samplers.set_samplers()
    assert resolver.cache_info().currsize == 0


def test_multi_sampler_reregistration_drops_cached_sampler_functions():
    import inspect

    sampling = types.SimpleNamespace(sample_euler=lambda model, x, **kwargs: x)
    multi = _definitions("extensions/openclaw-multi-sampler/scripts/openclaw_multi_sampler.py", {
        "Any": object, "inspect": inspect, "threading": threading,
        "k_diffusion": types.SimpleNamespace(sampling=sampling),
        "sd_samplers": types.SimpleNamespace(all_samplers=[], all_samplers_map={}, set_samplers=lambda: None),
        "sd_samplers_kdiffusion": types.SimpleNamespace(samplers_k_diffusion=[("Euler", "sample_euler")]),
        "_load_custom_defs": lambda: [],
    }, "_LOCK", "_REGISTERED_NAMES", "_TRANSIENT_DEFS", "_SIGNATURE_PARAM_CACHE", "_SAMPLER_FUNC_CACHE",
        "_signature_param_names", "_sampler_func_for", "_register_definitions")
    euler = multi["_sampler_func_for"]("Euler")[0]
    assert multi["_signature_param_names"](euler) == {"model", "x", "kwargs"}
    replacement = lambda model, x, **kwargs: x  # noqa: E731
    sampling.sample_euler = replacement
    assert multi["_sampler_func_for"]("Euler")[0] is euler  # cached until the chains are registered again

    multi["_register_definitions"]()

    # The id()-keyed signature cache must not outlive the functions it describes.
    assert multi["_SIGNATURE_PARAM_CACHE"] == {} and multi["_SAMPLER_FUNC_CACHE"] == {}
    assert multi["_sampler_func_for"]("Euler")[0] is replacement


def test_upscale_cache_keys_on_pixels_and_scaler_instance_returns_copies_and_evicts_lru():
    import hashlib
    from collections import OrderedDict

    from PIL import Image

    class Lock:
        held = 0

        def __enter__(self):
            self.held += 1

        def __exit__(self, *exc):
            self.held -= 1

    class GuardedCache(OrderedDict):
        def _guard(self):
            assert lock.held, "upscale cache used outside upscale_cache_lock"

        def get(self, *args):
            self._guard()
            return super().get(*args)

        def move_to_end(self, *args):
            self._guard()
            return super().move_to_end(*args)

        def clear(self):
            self._guard()
            return super().clear()

        def popitem(self, last=True):
            self._guard()
            return super().popitem(last)

        def __setitem__(self, key, value):
            self._guard()
            super().__setitem__(key, value)

    class Scaler:
        calls = 0

        def upscale(self, image, scale, data_path, *, target_size):
            Scaler.calls += 1
            return image.resize((image.width * scale, image.height * scale))

    lock, cache = Lock(), GuardedCache()
    fingerprints = []
    pixel_fingerprint = _definitions("modules/images.py", {"hashlib": hashlib}, "pixel_fingerprint")["pixel_fingerprint"]
    opts = types.SimpleNamespace(upscaling_max_images_in_cache=2)
    namespace = _definitions("scripts/postprocessing_upscale.py", {
        "os": os, "Image": Image,
        "images": types.SimpleNamespace(pixel_fingerprint=lambda image: (fingerprints.append(image), pixel_fingerprint(image))[1]),
        "shared": types.SimpleNamespace(opts=opts), "upscale_cache": cache, "upscale_cache_lock": lock,
    }, "_upscaler_identity", methods={"ScriptPostprocessingUpscale": {"upscale", "cached_upscale"}})
    script = namespace["ScriptPostprocessingUpscale"]()
    upscaler = types.SimpleNamespace(name="probe", data_path=None, scaler=Scaler())

    def upscale(image, upscaler=upscaler):
        return script.upscale(image, {}, upscaler, 0, 2, 0, None, None, False)

    def solid(color):
        return Image.new("RGB", (4, 4), color)

    red = (255, 0, 0)
    upscale(solid(red)).putpixel((0, 0), (0, 0, 0))  # the caller owns its result
    hit = upscale(solid(red))  # another image object with the same pixels
    assert Scaler.calls == 1 and hit.getpixel((0, 0)) == red
    hit.putpixel((0, 0), (0, 0, 0))
    assert upscale(solid(red)).getpixel((0, 0)) == red and Scaler.calls == 1

    upscale(solid((0, 255, 0)))
    assert Scaler.calls == 2
    upscale(solid(red))  # hit: red becomes the most recently used entry
    # The same model in another scaler instance is another entry; the third entry evicts green, the least recently used.
    upscale(solid(red), types.SimpleNamespace(name="probe", data_path=None, scaler=Scaler()))
    assert Scaler.calls == 3 and len(cache) == 2
    upscale(solid(red))
    assert Scaler.calls == 3
    upscale(solid((0, 255, 0)))
    assert Scaler.calls == 4

    # Cache size 0: no pixel hash, no copy (the upscaler's own result comes back), and lowering it drops the entries.
    opts.upscaling_max_images_in_cache = 0
    fingerprints.clear()
    results = []
    upscaler.scaler.upscale = lambda image, scale, data_path, target_size: results.append(image.resize((8, 8))) or results[-1]
    assert upscale(solid(red)) is results[-1]
    assert not fingerprints and not cache


def test_img2imgalt_reuses_noise_inversion_only_within_a_request_for_the_same_latent():
    import collections

    import torch

    inversions = []

    def find_noise(p, cond, uncond, cfg_scale, steps):
        inversions.append(p.init_latent)
        return torch.zeros_like(p.init_latent)

    def process_images(p):
        for latent in p.latents:
            p.init_latent = latent
            p.sample(None, None, [1], [0], 0.0, ["prompt"])

    sampler = types.SimpleNamespace(model_wrap=types.SimpleNamespace(get_sigmas=lambda steps: torch.ones(steps + 1)),
                                    sample_img2img=lambda p, x, noise, *args, **kwargs: noise)
    namespace = _definitions("scripts/img2imgalt.py", {
        "namedtuple": collections.namedtuple, "torch": torch,
        "prompt_parser": types.SimpleNamespace(SdConditioning=lambda prompts, **kwargs: prompts),
        "shared": types.SimpleNamespace(state=types.SimpleNamespace(job_count=0)),
        "processing": types.SimpleNamespace(process_images=process_images, create_random_tensors=lambda shape, **kwargs: torch.zeros(shape)),
        "sd_samplers": types.SimpleNamespace(create_sampler=lambda name, model: sampler),
        "find_noise_for_image": find_noise, "find_noise_for_image_sigma_adjustment": find_noise,
    }, "Cached", methods={"Script": {"__init__", "run"}})
    script = namespace["Script"]()

    def request(*latents):
        p = types.SimpleNamespace(
            latents=latents, init_latent=None, batch_size=1, width=64, height=64, sampler_name="Euler", steps=4, seed=1, subseed_strength=0.0,
            seed_resize_from_h=0, seed_resize_from_w=0, image_conditioning=None, extra_generation_params={},
            sd_model=types.SimpleNamespace(get_learned_conditioning=lambda prompts: prompts),
        )
        script.run(p, None, False, False, "prompt", "", False, 4, False, 1.0, 0.0, False)

    latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) / 8
    nudged = latent.clone()
    nudged[0, 0, 0, 0] += 0.5  # the former approximate comparison (summed x10 int64 difference < 100) took this for a hit
    request(latent, latent, nudged)
    assert [x is nudged for x in inversions] == [False, True]
    request(latent)  # a new request inverts again: nothing is retained on the script
    assert len(inversions) == 3


def test_hypertile_geometry_caches_are_bounded():
    hypertile = load_source("hypertile_cache_bounds", "extensions-builtin/hypertile/hypertile.py")
    for function in (hypertile.get_divisors, hypertile.find_hw_candidates):
        assert function.cache_info().maxsize == 256, function.__name__
