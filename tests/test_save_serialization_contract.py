import ast
import json
import types
from pathlib import Path
from types import SimpleNamespace


def load_function(path, name, globals_dict):
    source = Path(path).read_text(encoding="utf8")
    module = ast.parse(source, filename=path)
    function = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)
    compiled = compile(ast.Module(body=[function], type_ignores=[]), path, "exec")
    namespace = dict(globals_dict)
    exec(compiled, namespace)
    return namespace[name]


def test_processed_js_with_image_paths_keeps_paths_aligned_to_images():
    processed_js_with_image_paths = load_function(
        "modules/api/api.py",
        "processed_js_with_image_paths",
        {"json": json},
    )

    grid = SimpleNamespace()
    saved_sample = SimpleNamespace(already_saved_as="/tmp/sample-1.png")
    presaved_sample = SimpleNamespace(already_saved_as="/tmp/sample-2.png")
    processed = SimpleNamespace(
        images=[grid, saved_sample, presaved_sample],
        js=lambda: json.dumps({
            "index_of_first_image": 1,
            "infotexts": ["grid infotext", "sample 1 infotext"],
        }),
    )

    data = json.loads(processed_js_with_image_paths(processed, {"extra_key": "extra value"}))

    assert data["image_paths"] == [None, "/tmp/sample-1.png", "/tmp/sample-2.png"]
    assert data["index_of_first_image"] == 1
    assert data["infotexts"] == ["grid infotext", "sample 1 infotext"]
    assert data["extra_key"] == "extra value"


def make_save_files():
    saved_calls = []

    def parse_generation_parameters(text, _skip_fields):
        return {"Seed": text.split()[0], "Prompt": text, "Negative prompt": "negative"}

    def save_image(image, path, basename, seed, prompt, extension, info, grid, p, save_to_dirs):
        saved_calls.append({
            "image": image,
            "seed": seed,
            "prompt": prompt,
            "info": info,
            "grid": grid,
            "batch_index": p.batch_index,
            "save_to_dirs": save_to_dirs,
        })
        return f"{path}/{image}.{extension}", None

    class DummyContext:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    save_files = load_function(
        "modules/ui_common.py",
        "save_files",
        {
            "json": json,
            "os": types.SimpleNamespace(
                makedirs=lambda *args, **kwargs: None,
                path=types.SimpleNamespace(
                    join=lambda *parts: "/".join(parts),
                    exists=lambda _path: False,
                    relpath=lambda path, _start: path.rsplit("/", 1)[-1],
                    basename=lambda path: path.rsplit("/", 1)[-1],
                ),
            ),
            "csv": types.SimpleNamespace(writer=lambda _file: None),
            "nullcontext": DummyContext,
            "shared": SimpleNamespace(opts=SimpleNamespace(
                outdir_save="/tmp/out",
                use_save_to_dirs_for_ui=True,
                samples_format="png",
                save_selected_only=True,
                save_write_log_csv=False,
                grid_zip_filename_pattern="",
            )),
            "image_from_url_text": lambda filedata: filedata,
            "parameters_copypaste": SimpleNamespace(parse_generation_parameters=parse_generation_parameters),
            "modules": SimpleNamespace(images=SimpleNamespace(save_image=save_image)),
            "gr": SimpleNamespace(File=SimpleNamespace(update=lambda **kwargs: kwargs)),
            "plaintext_to_html": lambda text: text,
        },
    )

    return save_files, saved_calls


def generation_info(index_of_first_image=1):
    return json.dumps({
        "index_of_first_image": index_of_first_image,
        "infotexts": ["10 grid", "11 first sample", "12 second sample"],
        "width": 64,
        "height": 64,
        "sampler_name": "Euler",
        "cfg_scale": 7,
        "steps": 20,
        "sd_model_name": "model",
        "sd_model_hash": "hash",
    })


def test_save_files_save_all_uses_index_of_first_image_for_grid_prefix_and_infotexts():
    save_files, saved_calls = make_save_files()

    file_update, html = save_files(generation_info(), ["grid", "sample-1", "sample-2"], False, -1)

    assert [call["grid"] for call in saved_calls] == [True, False, False]
    assert [call["info"] for call in saved_calls] == ["10 grid", "11 first sample", "12 second sample"]
    assert [call["batch_index"] for call in saved_calls] == [-1, 0, 1]
    assert file_update == {"value": ["/tmp/out/grid.png", "/tmp/out/sample-1.png", "/tmp/out/sample-2.png"], "visible": True}
    assert html == "Saved: grid.png"


def test_save_files_save_all_without_grid_uses_sample_batch_indexes():
    save_files, saved_calls = make_save_files()

    file_update, html = save_files(generation_info(index_of_first_image=0), ["sample-1", "sample-2"], False, -1)

    assert [call["grid"] for call in saved_calls] == [False, False]
    assert [call["info"] for call in saved_calls] == ["10 grid", "11 first sample"]
    assert [call["batch_index"] for call in saved_calls] == [0, 1]
    assert file_update == {"value": ["/tmp/out/sample-1.png", "/tmp/out/sample-2.png"], "visible": True}
    assert html == "Saved: sample-1.png"


def test_save_files_save_selected_keeps_original_gallery_index_for_sample_metadata():
    save_files, saved_calls = make_save_files()

    file_update, html = save_files(generation_info(), ["grid", "sample-1", "sample-2"], False, 2)

    assert len(saved_calls) == 1
    assert saved_calls[0]["image"] == "sample-2"
    assert saved_calls[0]["grid"] is False
    assert saved_calls[0]["info"] == "12 second sample"
    assert saved_calls[0]["seed"] == "12"
    assert saved_calls[0]["batch_index"] == 1
    assert file_update == {"value": ["/tmp/out/sample-2.png"], "visible": True}
    assert html == "Saved: sample-2.png"
