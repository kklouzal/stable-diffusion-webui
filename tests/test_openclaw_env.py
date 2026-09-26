import pytest

from modules import openclaw_env


@pytest.mark.parametrize("value, expected", [("1", True), (" TRUE ", True), ("yes", True), ("On", True), ("0", False), ("false", False), ("No", False), ("off", False)])
def test_env_bool_grammar(monkeypatch, value, expected):
    monkeypatch.setenv("OPENCLAW_TEST_FLAG", value)
    assert openclaw_env.env_bool("OPENCLAW_TEST_FLAG", None) is expected


def test_env_bool_unset_or_empty_uses_default_and_rejects_other_values(monkeypatch):
    monkeypatch.delenv("OPENCLAW_TEST_FLAG", raising=False)
    assert openclaw_env.env_bool("OPENCLAW_TEST_FLAG", None) is None
    monkeypatch.setenv("OPENCLAW_TEST_FLAG", "  ")
    assert openclaw_env.env_bool("OPENCLAW_TEST_FLAG", True) is True
    monkeypatch.setenv("OPENCLAW_TEST_FLAG", "enable")
    with pytest.raises(ValueError, match="OPENCLAW_TEST_FLAG='enable' is not a boolean"):
        openclaw_env.env_bool("OPENCLAW_TEST_FLAG", False)


def test_env_int_grammar(monkeypatch):
    monkeypatch.delenv("OPENCLAW_TEST_INT", raising=False)
    assert openclaw_env.env_int("OPENCLAW_TEST_INT", 8, minimum=0) == 8
    monkeypatch.setenv("OPENCLAW_TEST_INT", "")
    assert openclaw_env.env_int("OPENCLAW_TEST_INT", 8, minimum=0) == 8
    monkeypatch.setenv("OPENCLAW_TEST_INT", " 0 ")
    assert openclaw_env.env_int("OPENCLAW_TEST_INT", 8, minimum=0) == 0
    monkeypatch.setenv("OPENCLAW_TEST_INT", "-7")
    with pytest.raises(ValueError, match="below the minimum 0"):
        openclaw_env.env_int("OPENCLAW_TEST_INT", 8, minimum=0)
    monkeypatch.setenv("OPENCLAW_TEST_INT", "not-an-int")
    with pytest.raises(ValueError, match="is not an integer"):
        openclaw_env.env_int("OPENCLAW_TEST_INT", 8, minimum=0)
