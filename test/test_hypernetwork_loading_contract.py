"""Hypernetwork activation (<hypernet:name:multiplier>): every mention applies with its own multiplier, and a mention
that cannot be loaded fails the request instead of generating without it."""
import pytest


@pytest.fixture
def hypernets(initialize, tmp_path, monkeypatch):
    from modules import shared
    from modules.hypernetworks import hypernetwork

    path = tmp_path / "net.pt"
    hypernetwork.Hypernetwork(name="net", enable_sizes=[8], layer_structure=[1, 2, 1], activation_func="linear", weight_init="Normal").save(str(path))
    (tmp_path / "net.pt.optim").write_bytes(b"not an optimizer state")  # inference never reads it
    (tmp_path / "broken.pt").write_bytes(b"not a hypernetwork")
    monkeypatch.setattr(shared, "hypernetworks", {"net": str(path), "broken": str(tmp_path / "broken.pt")}, raising=False)
    monkeypatch.setattr(shared, "loaded_hypernetworks", [], raising=False)
    monkeypatch.setattr(shared.opts, "print_hypernet_extra", False, raising=False)
    return hypernetwork, shared


def _multipliers(hypernetwork):
    return {layer.multiplier for layers in hypernetwork.layers.values() for layer in layers}


def test_each_mention_keeps_its_multiplier_and_shares_the_parsed_file(hypernets, monkeypatch):
    hypernetwork, shared = hypernets
    loads = []
    real_load = hypernetwork.Hypernetwork.load
    monkeypatch.setattr(hypernetwork.Hypernetwork, "load", lambda self, *args, **kwargs: loads.append(args) or real_load(self, *args, **kwargs))

    hypernetwork.load_hypernetworks(["net", "net"], [0.5, 1.5])

    first, second = shared.loaded_hypernetworks
    assert (_multipliers(first), _multipliers(second)) == ({0.5}, {1.5})  # the second mention overwrote the first
    assert first.layers[8][0].linear[0].weight is second.layers[8][0].linear[0].weight
    assert len(loads) == 1

    hypernetwork.load_hypernetworks(["net"], [0.25])  # reused from the previous activation
    assert len(loads) == 1 and [_multipliers(h) for h in shared.loaded_hypernetworks] == [{0.25}]


@pytest.mark.parametrize("name, message", [("missing", "hypernetwork not found: missing"), ("broken", "Error loading hypernetwork")])
def test_unloadable_mention_fails_instead_of_being_dropped(hypernets, name, message):
    hypernetwork, shared = hypernets
    hypernetwork.load_hypernetworks(["net"], [1.0])
    previous = list(shared.loaded_hypernetworks)

    with pytest.raises(RuntimeError, match=message):
        hypernetwork.load_hypernetworks(["net", name], [1.0, 1.0])

    assert shared.loaded_hypernetworks == previous
