class _WatchingDict(dict):
    """Records every mutation after which the watched key is missing."""

    def __init__(self, *args, watch):
        super().__init__(*args)
        self.watch = watch
        self.missing_after = []

    def _check(self, op):
        if self.watch not in self:
            self.missing_after.append(op)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._check("setitem")

    def __delitem__(self, key):
        super().__delitem__(key)
        self._check("delitem")

    def pop(self, *args):
        result = super().pop(*args)
        self._check("pop")
        return result

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._check("update")

    def clear(self):
        super().clear()
        self._check("clear")


def test_set_samplers_never_drops_a_registered_name(initialize, monkeypatch):
    from modules import sd_samplers

    watched = _WatchingDict(sd_samplers.samplers_map, watch="euler")
    monkeypatch.setattr(sd_samplers, "samplers_map", watched)

    sd_samplers.set_samplers()

    assert watched.missing_after == []
    assert watched["euler"] == "Euler"


def test_lookups_follow_the_published_registry(initialize, monkeypatch):
    from modules import sd_samplers, sd_samplers_common

    default = sd_samplers.get_sampler_and_scheduler("Registry Probe", None)[0]
    assert default != "Registry Probe"
    probe = sd_samplers_common.SamplerData("Registry Probe", lambda model: None, ["registry_probe"], {})
    monkeypatch.setattr(sd_samplers, "all_samplers", [*sd_samplers.all_samplers, probe])
    monkeypatch.setitem(sd_samplers.all_samplers_map, "Registry Probe", probe)
    try:
        sd_samplers.set_samplers()
        assert sd_samplers.get_sampler_and_scheduler("Registry Probe", None)[0] == "Registry Probe"
        assert sd_samplers.get_sampler_and_scheduler("registry_probe", None, strict=True)[0] == "Registry Probe"
    finally:
        monkeypatch.undo()
        sd_samplers.set_samplers()
    assert sd_samplers.get_sampler_and_scheduler("Registry Probe", None)[0] == default
