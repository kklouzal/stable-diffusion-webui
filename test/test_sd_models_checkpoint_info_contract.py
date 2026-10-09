import ast
import types


def load_path_is_parent():
    source = open("modules/util.py", encoding="utf8").read()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "path_is_parent")
    namespace = {"os": __import__("os")}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "modules/util.py", "exec"), namespace)
    return namespace["path_is_parent"]


def load_checkpoint_info_bits():
    source = open("modules/sd_models.py", encoding="utf8").read()
    tree = ast.parse(source)
    wanted = {"CheckpointInfo"}
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {
        "os": __import__("os"),
        "cache": types.SimpleNamespace(cached_data_for_file=lambda *args, **kwargs: {}),
        "errors": types.SimpleNamespace(display=lambda *args, **kwargs: None),
        "hashes": types.SimpleNamespace(partial_hash_from_cache=lambda filename: "abcd", sha256_from_cache=lambda filename, title: None),
        "model_path": "/models/Stable-diffusion",
        "read_metadata_from_safetensors": lambda filename: {},
        "shared": types.SimpleNamespace(cmd_opts=types.SimpleNamespace(ckpt_dir="/models/Stable-diffusion-extra")),
        "path_is_parent": load_path_is_parent(),
    }
    exec(compile(module, "modules/sd_models.py", "exec"), namespace)
    return namespace["CheckpointInfo"], namespace["path_is_parent"]


def test_checkpoint_info_uses_commonpath_for_model_roots():
    source = open("modules/sd_models.py", encoding="utf8").read()

    assert "path_is_parent = util.path_is_parent" in source
    assert "path_is_parent(abs_ckpt_dir, abspath)" in source
    assert "path_is_parent(model_path, abspath)" in source
    assert "abspath.startswith(abs_ckpt_dir)" not in source
    assert "abspath.startswith(model_path)" not in source


def test_checkpoint_info_rejects_sibling_prefix_model_path():
    CheckpointInfo, path_is_parent = load_checkpoint_info_bits()

    assert not path_is_parent("/models/Stable-diffusion", "/models/Stable-diffusion-extra/model.safetensors")
    assert path_is_parent("/models/Stable-diffusion-extra", "/models/Stable-diffusion-extra/model.safetensors")

    info = CheckpointInfo("/models/Stable-diffusion-extra/model.safetensors")

    assert info.name == "model.safetensors"
    assert info.model_name == "model"
