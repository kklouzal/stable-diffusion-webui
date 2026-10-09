from types import SimpleNamespace

import pytest

from modules import extra_networks


class RecordingNetwork(extra_networks.ExtraNetwork):
    def __init__(self, name, events, error=None, reset_error=None):
        super().__init__(name)
        self.events = events
        self.error = error
        self.reset_error = reset_error

    def activate(self, p, params_list):
        self.events.append((self.name, [params.items for params in params_list]))
        if params_list and self.error is not None:
            raise self.error
        if not params_list and self.reset_error is not None:
            raise self.reset_error

    def deactivate(self, p):
        self.events.append((self.name, "deactivate"))


class FatalLoraPreparationError(RuntimeError):
    pass


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(extra_networks, "extra_network_registry", {})
    monkeypatch.setattr(extra_networks, "extra_network_aliases", {})
    return extra_networks.extra_network_registry


def test_failing_requested_network_fails_the_request_instead_of_generating_without_it(registry):
    events = []
    extra_networks.register_extra_network(RecordingNetwork("hypernet", events, error=ValueError("could not convert string to float: 'x'")))
    extra_networks.register_extra_network(RecordingNetwork("lora", events))
    _, data = extra_networks.parse_prompt("a cat <hypernet:style:x>")

    with pytest.raises(ValueError, match="could not convert"):
        extra_networks.activate(SimpleNamespace(scripts=None), data)

    # Before the fix the error was only displayed and the reset pass re-activated hypernet with no networks.
    assert ("hypernet", []) not in events


def test_fatal_lora_preparation_error_still_propagates(registry):
    events = []
    extra_networks.register_extra_network(RecordingNetwork("lora", events, error=FatalLoraPreparationError("stale quantized state")))
    _, data = extra_networks.parse_prompt("a cat <lora:detail:0.5>")

    with pytest.raises(FatalLoraPreparationError):
        extra_networks.activate(SimpleNamespace(scripts=None), data)


def test_successful_activation_resets_unmentioned_networks(registry):
    events = []
    extra_networks.register_extra_network(RecordingNetwork("hypernet", events))
    extra_networks.register_extra_network(RecordingNetwork("lora", events))
    _, data = extra_networks.parse_prompt("a cat <lora:detail:0.5>")

    extra_networks.activate(SimpleNamespace(scripts=None), data)

    assert events == [("lora", [["detail", "0.5"]]), ("hypernet", [])]


def test_failing_reset_of_an_unmentioned_network_fails_the_request(registry):
    # Any network, not only LoRA: a hypernet reset can append opts.sd_hypernetwork to the prompts and then fail, which
    # would name a hypernetwork in infotext that was never applied.
    events = []
    extra_networks.register_extra_network(RecordingNetwork("hypernet", events, reset_error=OSError("missing hypernetwork")))
    extra_networks.register_extra_network(RecordingNetwork("lora", events))
    _, data = extra_networks.parse_prompt("a cat <lora:detail:0.5>")

    with pytest.raises(OSError, match="missing hypernetwork"):
        extra_networks.activate(SimpleNamespace(scripts=None), data)

    assert events == [("lora", [["detail", "0.5"]]), ("hypernet", [])]
