from pathlib import Path


def test_set_theme_uses_url_search_params():
    source = Path("javascript/ui.js").read_text()

    assert "function make_theme_url(url, theme)" in source
    assert "new URL(url)" in source
    assert "gradioURL.searchParams.has('__theme')" in source
    assert "gradioURL.searchParams.set('__theme', theme)" in source
    assert "gradioURL + '?__theme=' + theme" not in source


def test_extra_networks_toggle_css_clears_existing_style():
    source = Path("javascript/extraNetworks.js").read_text()

    assert "style.textContent = '';" in source
    assert "style.innerHTML == '';" not in source
