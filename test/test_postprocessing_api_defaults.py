import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


def load_run_extras():
    source = Path("modules/postprocessing.py").read_text()
    tree = ast.parse(source)
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_extras"],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/postprocessing.py", "exec"), namespace)
    return namespace["run_extras"]


def load_run_postprocessing():
    source = Path("modules/postprocessing.py").read_text()
    tree = ast.parse(source)
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_postprocessing"],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/postprocessing.py", "exec"), namespace)
    return namespace["run_postprocessing"]


def test_postprocessing_runner_order_override_preserves_script_defaults(monkeypatch):
    from modules import scripts_postprocessing

    calls = []

    def make_script(name, order, args_from, default):
        script = SimpleNamespace(
            name=name,
            order=order,
            main_ui_only=False,
            controls=OrderedDict([("value", SimpleNamespace(value=default))]),
            args_from=args_from,
            args_to=args_from + 1,
        )
        script.process_firstpass = lambda _pp, **_kwargs: calls.append(("first", name))
        script.process = lambda _pp, **_kwargs: calls.append(("process", name))
        return script

    monkeypatch.setattr(
        scripts_postprocessing.shared,
        "opts",
        SimpleNamespace(postprocessing_operation_order=[], postprocessing_disable_in_extras=[]),
    )
    monkeypatch.setattr(scripts_postprocessing.shared, "state", SimpleNamespace(skipped=False, job=None), raising=False)

    runner = scripts_postprocessing.ScriptPostprocessingRunner()
    runner.scripts = [
        make_script("Upscale", 1000, 0, "upscale-default"),
        make_script("GFPGAN", 2000, 1, "gfpgan-default"),
    ]
    runner.ui_created = True

    args = runner.create_args_for_run({"GFPGAN": {"value": "requested"}}, scripts_order=["GFPGAN", "Upscale"])
    assert args == ["upscale-default", "requested"]

    pp = scripts_postprocessing.PostprocessedImage(SimpleNamespace())
    runner.run(pp, args, scripts_order=["GFPGAN", "Upscale"])

    assert calls == [
        ("first", "GFPGAN"),
        ("first", "Upscale"),
        ("process", "GFPGAN"),
        ("process", "Upscale"),
    ]


def test_ui_batch_upload_skips_unreadable_files_before_job_count():
    from PIL import Image

    run_postprocessing = load_run_postprocessing()
    valid_image = Image.new("RGB", (1, 1))

    class FakeState:
        interrupted = False
        stopping_generation = False
        skipped = False
        job = None
        textinfo = None

        def __init__(self):
            self.job_count = None
            self.nextjob_calls = 0

        def begin(self, job):
            self.job = job

        def nextjob(self):
            self.nextjob_calls += 1

        def assign_current_image(self, image):
            self.current_image = image

        def end(self):
            self.ended = True

    class FakePostprocessedImage:
        def __init__(self, image):
            self.image = image
            self.extra_images = []
            self.info = {}
            self.caption = None

        def get_suffix(self, used_suffixes):
            return ""

    state = FakeState()

    def fake_read(path):
        if path.endswith("bad.png"):
            raise RuntimeError("unreadable")
        return valid_image.copy()

    run_postprocessing.__globals__.update(
        Image=Image,
        os=__import__("os"),
        devices=SimpleNamespace(torch_gc=lambda: None),
        images=SimpleNamespace(
            read=fake_read,
            fix_image=lambda image: image,
            read_info_from_image=lambda image: ("", {}),
        ),
        scripts_postprocessing=SimpleNamespace(PostprocessedImage=FakePostprocessedImage),
        scripts=SimpleNamespace(scripts_postproc=SimpleNamespace(run=lambda pp, args, scripts_order=None: None)),
        opts=SimpleNamespace(
            outdir_samples="",
            outdir_extras_samples="",
            use_original_name_batch=False,
            enable_pnginfo=False,
            samples_format="png",
        ),
        shared=SimpleNamespace(
            state=state,
            cmd_opts=SimpleNamespace(hide_ui_dir_config=False),
            listfiles=lambda input_dir: [],
            opts=SimpleNamespace(postprocessing_existing_caption_action="Ignore"),
        ),
        ui_common=SimpleNamespace(plaintext_to_html=lambda text: text),
        infotext_utils=SimpleNamespace(quote=lambda value: value),
    )

    bad_upload = SimpleNamespace(name="/tmp/bad.png", orig_name="bad.png")
    good_upload = SimpleNamespace(name="/tmp/good.png", orig_name="good.png")

    outputs, html_info, html_log = run_postprocessing(1, None, [bad_upload, good_upload], "", "", True, save_output=False)

    assert len(outputs) == 1
    assert state.job_count == 1
    assert state.nextjob_calls == 1
    assert html_info == ""
    assert html_log == ""


def test_run_extras_maps_upscale_first_to_postprocessing_order():
    run_extras = load_run_extras()
    observed = []

    class FakeRunner:
        def create_args_for_run(self, scripts_args, scripts_order=None):
            observed.append(("create", scripts_order, scripts_args))
            return ["arg"]

    def fake_run_postprocessing(*args, save_output=True, scripts_order=None):
        observed.append(("run", scripts_order, save_output, args[-1]))
        return "result"

    run_extras.__globals__["scripts"] = SimpleNamespace(scripts_postproc=FakeRunner())
    run_extras.__globals__["run_postprocessing"] = fake_run_postprocessing

    common_args = dict(
        extras_mode=0,
        resize_mode=0,
        image="image",
        image_folder="",
        input_dir="",
        output_dir="",
        show_extras_results=True,
        gfpgan_visibility=0.5,
        codeformer_visibility=0.25,
        codeformer_weight=0.75,
        upscaling_resize=2,
        upscaling_resize_w=512,
        upscaling_resize_h=512,
        upscaling_crop=True,
        extras_upscaler_1="Lanczos",
        extras_upscaler_2="None",
        extras_upscaler_2_visibility=0,
    )

    assert run_extras(**common_args, upscale_first=False) == "result"
    assert observed[0][1] == ["GFPGAN", "CodeFormer", "Upscale"]
    assert observed[1][1] == ["GFPGAN", "CodeFormer", "Upscale"]

    observed.clear()
    assert run_extras(**common_args, upscale_first=True) == "result"
    assert observed[0][1] == ["Upscale", "GFPGAN", "CodeFormer"]
    assert observed[1][1] == ["Upscale", "GFPGAN", "CodeFormer"]


def load_set_upscalers():
    source = Path("modules/api/api.py").read_text()
    tree = ast.parse(source)
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "setUpscalers"],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/api/api.py", "exec"), namespace)
    return namespace["setUpscalers"]


def load_api_extras_helpers():
    source = Path("modules/api/api.py").read_text()
    tree = ast.parse(source)
    names = {"decode_extras_batch_images"}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "modules/api/api.py", "exec"), namespace)
    return namespace


def load_api_extras_method(method_name):
    source = Path("modules/api/api.py").read_text()
    tree = ast.parse(source)
    api_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    fake_class = ast.ClassDef(name="FakeApi", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.Module(body=[fake_class], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"models": SimpleNamespace(ExtrasSingleImageRequest=object, ExtrasBatchImagesRequest=object)}
    exec(compile(module, "modules/api/api.py", "exec"), namespace)
    return namespace["FakeApi"]


def test_api_extras_always_returns_images_despite_directory_gallery_toggle():
    set_upscalers = load_set_upscalers()
    req = SimpleNamespace(
        show_extras_results=False,
        upscaler_1="None",
        upscaler_2="None",
    )

    result = set_upscalers(req)

    assert result["show_extras_results"] is True
    assert result["extras_upscaler_1"] == "None"
    assert result["extras_upscaler_2"] == "None"
    assert "upscaler_1" not in result
    assert "upscaler_2" not in result


def test_extras_batch_decode_skips_corrupt_images_but_keeps_valid_items():
    helpers = load_api_extras_helpers()
    invalid_image_error = RuntimeError("invalid")
    invalid_image_error.detail = "Invalid encoded image"

    def fake_decode(data):
        if data == "bad":
            raise invalid_image_error
        return f"decoded:{data}"

    helpers["decode_base64_to_image"] = fake_decode
    helpers["HTTPException"] = RuntimeError

    result = helpers["decode_extras_batch_images"]([
        SimpleNamespace(data="good-1"),
        SimpleNamespace(data="bad"),
        SimpleNamespace(data="good-2"),
    ])

    assert result == ["decoded:good-1", "decoded:good-2"]


def test_extras_single_response_allows_no_output_from_skipped_or_interrupted_run():
    api_class = load_api_extras_method("extras_single_image_api")
    api = api_class()

    class Lock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_set_upscalers(req):
        return {"image": req.image}

    api.queue_lock = Lock()
    api.extras_single_image_api.__globals__.update(
        setUpscalers=fake_set_upscalers,
        decode_base64_to_image=lambda image: image,
        encode_pil_to_base64=lambda image: (_ for _ in ()).throw(AssertionError("no image should be encoded")),
        postprocessing=SimpleNamespace(run_extras=lambda **kwargs: ([], "info", "")),
        models=SimpleNamespace(ExtrasSingleImageResponse=lambda **kwargs: kwargs),
    )

    assert api.extras_single_image_api(SimpleNamespace(image="input")) == {"image": None, "html_info": "info"}


def test_run_postprocessing_honors_skip_during_extra_image_output_loop():
    from PIL import Image

    run_postprocessing = load_run_postprocessing()
    source = Image.new("RGB", (1, 1), color="white")
    extra = Image.new("RGB", (1, 1), color="black")

    class FakeState:
        interrupted = False
        stopping_generation = False
        skipped = False
        job = None
        textinfo = None

        def begin(self, job):
            self.job = job

        def nextjob(self):
            pass

        def assign_current_image(self, image):
            self.current_image = image
            self.skipped = True

        def end(self):
            self.ended = True

    class FakePostprocessedImage:
        def __init__(self, image):
            self.image = image
            self.extra_images = []
            self.info = {}
            self.caption = None

        def get_suffix(self, used_suffixes):
            return ""

    state = FakeState()

    def fake_run(pp, args, scripts_order=None):
        pp.extra_images.append(FakePostprocessedImage(extra))

    run_postprocessing.__globals__.update(
        Image=Image,
        os=__import__("os"),
        devices=SimpleNamespace(torch_gc=lambda: None),
        images=SimpleNamespace(
            fix_image=lambda image: image,
            read_info_from_image=lambda image: ("", {}),
        ),
        scripts_postprocessing=SimpleNamespace(PostprocessedImage=FakePostprocessedImage),
        scripts=SimpleNamespace(scripts_postproc=SimpleNamespace(run=fake_run)),
        opts=SimpleNamespace(
            outdir_samples="",
            outdir_extras_samples="",
            use_original_name_batch=False,
            enable_pnginfo=False,
            samples_format="png",
        ),
        shared=SimpleNamespace(
            state=state,
            cmd_opts=SimpleNamespace(hide_ui_dir_config=False),
            listfiles=lambda input_dir: [],
            opts=SimpleNamespace(postprocessing_existing_caption_action="Ignore"),
        ),
        ui_common=SimpleNamespace(plaintext_to_html=lambda text: text),
        infotext_utils=SimpleNamespace(quote=lambda value: value),
    )

    outputs, html_info, html_log = run_postprocessing(0, source, [], "", "", True, save_output=False)

    assert outputs == [source]
    assert state.current_image is source
    assert state.skipped is True
    assert state.ended is True
    assert html_info == ""
    assert html_log == ""
