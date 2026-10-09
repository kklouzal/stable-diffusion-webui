# ControlNet (repo-owned copy)

API-only ControlNet for this A1111 fork. It was vendored from
[Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) v1.1.455 and is maintained here; the
upstream browser UI, batch/AnimateDiff/SparseCtrl/PuLID paths, movie2movie script, installer and offline converter
CLIs were removed. `gb10/run.sh` mirrors this directory into the deployment's extensions directory.

## API
- Generation: `alwayson_scripts.ControlNet.args` is a list of `ControlNetUnit` objects
  (`internal_controlnet/args.py`; `/sdapi/v1/script-info` reports the default unit). Requests with more units than
  `control_net_unit_count` slots use all of them.
- Legacy top-level `control_net_*` request fields (and their `2`/`3` suffixed variants) configure the units when
  the `control_net_allow_script_control` option is on.
- Routes (`scripts/api.py`): `/controlnet/version`, `model_list`, `module_list`, `control_types`, `settings`,
  `detect`, `render_openpose_json`.

## Models and annotator weights
- ControlNet models: `models/` in this directory, `<models>/ControlNet`, `--controlnet-dir` and the
  `control_net_models_path` option. `models/*.sha256` record the deployed weights.
- Preprocessor (annotator) weights download on first use to `annotator/downloads/`, or to the directory given by
  `control_net_preprocessor_models_path` / `control_net_modules_path` / `--controlnet-annotator-models-path`.

## Command-line flags (`preload.py`)
`--controlnet-dir`, `--controlnet-annotator-models-path`, `--no-half-controlnet`,
`--controlnet-preprocessor-cache-size` (default 16), `--controlnet-loglevel`, `--controlnet-tracemalloc`.

## Tests
See `tests/README.md`.

## License
GPL-3.0 (`LICENSE`). Vendored annotators keep their own `LICENSE` files.
