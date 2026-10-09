import os


def test_create_hypernetwork_returns_created_filename_for_ui_message(initialize, tmp_path, monkeypatch):
    from modules import shared
    from modules.hypernetworks import hypernetwork

    monkeypatch.setattr(shared.cmd_opts, "hypernetwork_dir", str(tmp_path))
    reloads = []
    monkeypatch.setattr(shared, "reload_hypernetworks", lambda: reloads.append(True))

    filename = hypernetwork.create_hypernetwork("probe/net", ["8"], False, layer_structure="1, 2, 1", activation_func="linear", weight_init="Normal")

    assert filename == os.path.join(str(tmp_path), "probenet.pt")
    assert os.path.isfile(filename)
    assert reloads == [True]


def test_attention_projection_boundary_normalizes_strided_context(initialize):
    # Quantized linear backends flatten the projections' input with view(), which needs contiguous storage.
    import torch

    from modules.hypernetworks import hypernetwork

    context = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(2, 8, 4).transpose(1, 2)
    assert not context.is_contiguous()

    context_k, context_v = hypernetwork.apply_hypernetworks([], context)

    assert context_k.is_contiguous() and context_v.is_contiguous()
    assert torch.equal(context_k, context) and torch.equal(context_v, context)
