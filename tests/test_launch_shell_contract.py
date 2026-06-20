from pathlib import Path


def test_webui_accelerate_flag_requires_true_value():
    source = Path("webui.sh").read_text()

    assert '[[ "${ACCELERATE}" == "True" ]]' in source
    assert '[ ${ACCELERATE}="True" ]' not in source
