import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def load_script_listing_api_class():
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    module = ast.parse(source)
    api_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    wanted = {"_script_lists", "get_scripts_list", "get_script_info"}
    methods = [node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(subset)

    namespace = {}
    exec(compile(subset, "<api-script-listing>", "exec"), namespace)
    return namespace["Api"]


def test_script_listing_endpoints_share_txt2img_img2img_sources(monkeypatch):
    api_class = load_script_listing_api_class()

    txt_info = object()
    img_info = object()
    txt2img_scripts = [
        SimpleNamespace(name="Selectable", api_info=txt_info),
        SimpleNamespace(name=None, api_info=None),
    ]
    img2img_scripts = [SimpleNamespace(name="Img Script", api_info=img_info)]

    scripts_stub = SimpleNamespace(
        scripts_txt2img=SimpleNamespace(scripts=txt2img_scripts),
        scripts_img2img=SimpleNamespace(scripts=img2img_scripts),
    )
    scripts_list_calls = []

    class ScriptsList(SimpleNamespace):
        def __init__(self, **kwargs):
            scripts_list_calls.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setitem(api_class._script_lists.__globals__, "scripts", scripts_stub)
    monkeypatch.setitem(api_class.get_scripts_list.__globals__, "models", SimpleNamespace(ScriptsList=ScriptsList))

    api = api_class()

    scripts_list = api.get_scripts_list()
    assert scripts_list.txt2img == ["Selectable"]
    assert scripts_list.img2img == ["Img Script"]
    assert scripts_list_calls == [{"txt2img": ["Selectable"], "img2img": ["Img Script"]}]
    assert api.get_script_info() == [txt_info, img_info]


def load_latent_upscale_api_class():
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    module = ast.parse(source)
    api_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_latent_upscale_modes")
    subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(subset)

    namespace = {}
    exec(compile(subset, "<api-latent-upscale-modes>", "exec"), namespace)
    return namespace["Api"]


def load_metadata_listing_api_class():
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    module = ast.parse(source)
    api_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    wanted = {
        "get_config",
        "get_cmd_flags",
        "get_samplers",
        "get_schedulers",
        "get_upscalers",
        "get_sd_models",
        "get_sd_vaes",
        "get_hypernetworks",
        "get_embeddings",
    }
    methods = [node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(subset)

    namespace = {}
    exec(compile(subset, "<api-metadata-listing>", "exec"), namespace)
    return namespace["Api"]


def test_config_and_flags_preserve_public_mapping_sources(monkeypatch):
    api_class = load_metadata_listing_api_class()
    shared_stub = SimpleNamespace(
        opts=SimpleNamespace(
            data={
                "saved": "custom",
                "defaulted": None,
                "dynamic_only": "runtime",
            },
            data_labels={
                "saved": SimpleNamespace(default="fallback"),
                "defaulted": SimpleNamespace(default="default-value"),
            },
        ),
        cmd_opts=SimpleNamespace(api=True, port=7860),
    )
    monkeypatch.setitem(api_class.get_config.__globals__, "shared", shared_stub)
    monkeypatch.setitem(api_class.get_cmd_flags.__globals__, "shared", shared_stub)

    assert api_class().get_config() == {
        "saved": "custom",
        "defaulted": None,
        "dynamic_only": "runtime",
    }
    assert api_class().get_cmd_flags() == {"api": True, "port": 7860}


def test_model_metadata_listing_shapes_match_response_models(monkeypatch):
    api_class = load_metadata_listing_api_class()
    api_globals = api_class.get_samplers.__globals__
    api_globals["sd_samplers"] = SimpleNamespace(all_samplers=[("Euler", "unused", ["euler"], {"scheduler": "normal"})])
    api_globals["sd_schedulers"] = SimpleNamespace(
        schedulers=[
            SimpleNamespace(
                name="karras",
                label="Karras",
                aliases=["kar"],
                default_rho=7.0,
                need_inner_model=False,
            )
        ]
    )
    api_globals["shared"] = SimpleNamespace(
        sd_upscalers=[
            SimpleNamespace(
                name="Lanczos",
                scaler=SimpleNamespace(model_name=None),
                data_path=None,
                scale=4.0,
            )
        ],
        hypernetworks={"hyp": "/models/hyp.pt"},
    )
    api_globals["find_checkpoint_config_near_filename"] = lambda checkpoint: checkpoint.config_path

    sd_models_stub = types.ModuleType("modules.sd_models")
    sd_models_stub.checkpoints_list = {
        "ckpt": SimpleNamespace(
            title="Model Title",
            model_name="model",
            shorthash="abc123",
            sha256="sha256",
            filename="/models/model.safetensors",
            config_path="/models/model.yaml",
        )
    }
    sd_vae_stub = types.ModuleType("modules.sd_vae")
    sd_vae_stub.vae_dict = {"vae": "/models/vae.pt"}
    modules_stub = types.ModuleType("modules")
    modules_stub.sd_models = sd_models_stub
    modules_stub.sd_vae = sd_vae_stub
    monkeypatch.setitem(sys.modules, "modules", modules_stub)
    monkeypatch.setitem(sys.modules, "modules.sd_models", sd_models_stub)
    monkeypatch.setitem(sys.modules, "modules.sd_vae", sd_vae_stub)

    api = api_class()

    assert api.get_samplers() == [{"name": "Euler", "aliases": ["euler"], "options": {"scheduler": "normal"}}]
    assert api.get_schedulers() == [
        {
            "name": "karras",
            "label": "Karras",
            "aliases": ["kar"],
            "default_rho": 7.0,
            "need_inner_model": False,
        }
    ]
    assert api.get_upscalers() == [
        {
            "name": "Lanczos",
            "model_name": None,
            "model_path": None,
            "model_url": None,
            "scale": 4.0,
        }
    ]
    assert api.get_sd_models() == [
        {
            "title": "Model Title",
            "model_name": "model",
            "hash": "abc123",
            "sha256": "sha256",
            "filename": "/models/model.safetensors",
            "config": "/models/model.yaml",
        }
    ]
    assert api.get_sd_vaes() == [{"model_name": "vae", "filename": "/models/vae.pt"}]
    assert api.get_hypernetworks() == [{"name": "hyp", "path": "/models/hyp.pt"}]


def test_embeddings_response_preserves_loaded_and_skipped_maps(monkeypatch):
    api_class = load_metadata_listing_api_class()
    loaded = SimpleNamespace(
        name="loaded-token",
        step=10,
        sd_checkpoint="abc",
        sd_checkpoint_name="checkpoint",
        shape=768,
        vectors=2,
    )
    skipped = SimpleNamespace(
        name="skipped-token",
        step=None,
        sd_checkpoint=None,
        sd_checkpoint_name=None,
        shape=1024,
        vectors=1,
    )
    embedding_db = SimpleNamespace(
        word_embeddings={"loaded-token": loaded},
        skipped_embeddings={"skipped-token": skipped},
    )
    sd_hijack_stub = SimpleNamespace(model_hijack=SimpleNamespace(embedding_db=embedding_db))
    monkeypatch.setitem(api_class.get_embeddings.__globals__, "sd_hijack", sd_hijack_stub)

    assert api_class().get_embeddings() == {
        "loaded": {
            "loaded-token": {
                "step": 10,
                "sd_checkpoint": "abc",
                "sd_checkpoint_name": "checkpoint",
                "shape": 768,
                "vectors": 2,
            }
        },
        "skipped": {
            "skipped-token": {
                "step": None,
                "sd_checkpoint": None,
                "sd_checkpoint_name": None,
                "shape": 1024,
                "vectors": 1,
            }
        },
    }


def test_latent_upscale_modes_lists_shared_mode_names_in_order(monkeypatch):
    api_class = load_latent_upscale_api_class()
    shared_stub = SimpleNamespace(
        latent_upscale_modes={
            "Latent": {"mode": "bilinear", "antialias": False},
            "Latent (nearest)": {"mode": "nearest", "antialias": False},
        }
    )

    monkeypatch.setitem(api_class.get_latent_upscale_modes.__globals__, "shared", shared_stub)

    assert api_class().get_latent_upscale_modes() == [
        {"name": "Latent"},
        {"name": "Latent (nearest)"},
    ]


def load_extension_listing_api_class():
    source = Path("modules/api/api.py").read_text(encoding="utf8")
    module = ast.parse(source)
    api_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Api")
    method = next(node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name == "get_extensions_list")
    subset = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(subset)

    namespace = {}
    exec(compile(subset, "<api-extension-listing>", "exec"), namespace)
    return namespace["Api"]


def test_extensions_list_preserves_public_git_metadata_shape(monkeypatch):
    api_class = load_extension_listing_api_class()
    events = []

    class Extension:
        pass

    class ExtensionRecord:
        def __init__(self, name, remote, enabled=True, path="/extensions/private-path"):
            self.name = name
            self.path = path
            self.remote = remote
            self.branch = None
            self.commit_hash = "abc123"
            self.commit_date = None
            self.version = "abc123"
            self.enabled = enabled

        def read_info_from_repo(self):
            events.append(("read", self.name))

    remote_extension = ExtensionRecord("remote-ext", "https://example.invalid/repo.git", enabled=False)
    local_extension = ExtensionRecord("local-ext", None)

    def list_extensions():
        events.append(("list", None))

    extensions_stub = types.ModuleType("modules.extensions")
    extensions_stub.Extension = Extension
    extensions_stub.extensions = [remote_extension, local_extension]
    extensions_stub.list_extensions = list_extensions

    modules_stub = types.ModuleType("modules")
    modules_stub.extensions = extensions_stub
    monkeypatch.setitem(sys.modules, "modules", modules_stub)
    monkeypatch.setitem(sys.modules, "modules.extensions", extensions_stub)

    assert api_class().get_extensions_list() == [
        {
            "name": "remote-ext",
            "remote": "https://example.invalid/repo.git",
            "branch": None,
            "commit_hash": "abc123",
            "commit_date": None,
            "version": "abc123",
            "enabled": False,
        }
    ]
    assert events == [("list", None), ("read", "remote-ext"), ("read", "local-ext")]
