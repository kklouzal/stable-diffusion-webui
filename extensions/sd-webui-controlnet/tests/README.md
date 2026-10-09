# Tests

Backend tests that import the A1111 modules in-process (no server, no GPU). `tests/utils.py` initializes A1111
(without the startup model load) when a test module imports it, so every test module imports it first.

- `cn_script/`: the ControlNet script, hooks, preprocessor cache and model loading.
- `annotator_tests/`: annotator output and pre/post-processing fixes.
- `external_code_api/`: the `internal_controlnet.external_code` helpers.
- `../unit_tests/`: `ControlNetUnit` validation, run by the repository test
  `tests/test_controlnet_legacy_api_fields.py`.

Run them from the A1111 root inside the deploy image (CPU only), for example:

```shell
python -m pytest -q -p no:cacheprovider extensions/sd-webui-controlnet/tests
```
