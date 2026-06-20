from types import SimpleNamespace

from modules.ui_loadsave import UiLoadsave


def test_ui_defaults_review_escapes_html_values():
    loadsave = UiLoadsave.__new__(UiLoadsave)
    loadsave.component_mapping = {"bad/<path>": SimpleNamespace(choices=[])}
    loadsave.read_from_file = lambda: {"bad/<path>": "<b>old</b>"}

    html = loadsave.ui_view("<script>alert(1)</script>")

    assert "bad/&lt;path&gt;" in html
    assert "&lt;b&gt;old&lt;/b&gt;" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<b>old</b>" not in html
    assert "<script>alert(1)</script>" not in html
