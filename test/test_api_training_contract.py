import ast
from pathlib import Path


class _Lock:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("lock-enter")

    def __exit__(self, *exc):
        self.events.append("lock-exit")


def _training_api(events, fail=None):
    """The create/train endpoints of Api and their task helpers, run against recording stand-ins of the model code.

    fail names the stand-in that raises: "create" (AssertionError, as create_* validation does), "train" or
    "restore" (the hypernetwork's after-training device restore)."""
    from types import SimpleNamespace

    source = Path("modules/api/api.py").read_text(encoding="utf8")
    api_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Api")
    wanted = {
        "_create_response", "_train_response", "_run_create_task", "_run_training_task",
        "create_embedding", "create_hypernetwork", "train_embedding", "train_hypernetwork",
        "_prepare_hypernetwork_training", "_restore_hypernetwork_training_devices",
    }
    methods = [node for node in api_class.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    module = ast.Module(body=[ast.ClassDef(name="Api", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])

    def create(kind):
        def create_fn(**args):
            events.append((kind, args))
            if fail == "create":
                raise AssertionError("name taken")
            return f"/{kind}.pt"
        return create_fn

    def train(kind):
        def train_fn(**args):
            events.append((kind, args, len(shared.loaded_hypernetworks)))
            if fail == "train":
                raise RuntimeError("diverged")
            return object(), f"/{kind}.pt"
        return train_fn

    def to(name):
        def move(device):
            if fail == "restore":
                raise RuntimeError("device lost")
            events.append(("to", name, device))
        return move

    shared = SimpleNamespace(
        state=SimpleNamespace(begin=lambda job: events.append(("begin", job)), end=lambda: events.append("end")),
        opts=SimpleNamespace(training_xattention_optimizations=False),
        loaded_hypernetworks=["previous"],
        sd_model=SimpleNamespace(cond_stage_model=SimpleNamespace(to=to("cond_stage_model")), first_stage_model=SimpleNamespace(to=to("first_stage_model"))),
    )
    namespace = {
        "models": SimpleNamespace(CreateResponse=lambda info: ("create", info), TrainResponse=lambda info: ("train", info)),
        "shared": shared,
        "devices": SimpleNamespace(device="gpu"),
        "sd_hijack": SimpleNamespace(
            undo_optimizations=lambda: events.append("undo-optimizations"),
            apply_optimizations=lambda: events.append("apply-optimizations"),
            model_hijack=SimpleNamespace(embedding_db=SimpleNamespace(load_textual_inversion_embeddings=lambda: events.append("reload-embeddings"))),
        ),
        "create_embedding": create("create_embedding"),
        "create_hypernetwork": create("create_hypernetwork"),
        "train_embedding": train("train_embedding"),
        "train_hypernetwork": train("train_hypernetwork"),
    }
    exec(compile(ast.fix_missing_locations(module), "modules/api/api.py", "exec"), namespace)
    api = namespace["Api"]()
    api.queue_lock = _Lock(events)
    return api


def test_create_endpoints_report_created_filename_under_queue_lock():
    events = []
    api = _training_api(events)
    assert api.create_embedding({"name": "e"}) == ("create", "create embedding filename: /create_embedding.pt")
    assert events == ["lock-enter", ("begin", "create_embedding"), ("create_embedding", {"name": "e"}), "reload-embeddings", "end", "lock-exit"]

    events.clear()
    assert api.create_hypernetwork({"name": "h"}) == ("create", "create hypernetwork filename: /create_hypernetwork.pt")
    assert events == ["lock-enter", ("begin", "create_hypernetwork"), ("create_hypernetwork", {"name": "h"}), "end", "lock-exit"]

    events = []
    api = _training_api(events, fail="create")
    assert api.create_embedding({"name": "e"}) == ("create", "create embedding error: name taken")
    assert api.create_hypernetwork({"name": "h"}) == ("create", "create hypernetwork error: name taken")
    assert "reload-embeddings" not in events
    assert events.count("end") == 2


def test_train_endpoints_report_their_own_kind_and_end_state_once():
    events = []
    api = _training_api(events)
    assert api.train_hypernetwork({"steps": 1}) == ("train", "train hypernetwork complete: filename: /train_hypernetwork.pt error: None")
    assert events == [
        "lock-enter",
        ("begin", "train_hypernetwork"),
        "undo-optimizations",
        ("train_hypernetwork", {"steps": 1}, 0),  # loaded hypernetworks are dropped before training
        ("to", "cond_stage_model", "gpu"),
        ("to", "first_stage_model", "gpu"),
        "apply-optimizations",
        "end",
        "lock-exit",
    ]

    events.clear()
    assert api.train_embedding({"steps": 1}) == ("train", "train embedding complete: filename: /train_embedding.pt error: None")
    assert events == ["lock-enter", ("begin", "train_embedding"), "undo-optimizations", ("train_embedding", {"steps": 1}, 0), "apply-optimizations", "end", "lock-exit"]

    # A training failure is reported in the completion message; the devices and optimizations are still restored.
    events = []
    api = _training_api(events, fail="train")
    assert api.train_hypernetwork({"steps": 1}) == ("train", "train hypernetwork complete: filename:  error: diverged")
    assert events[-4:] == [("to", "first_stage_model", "gpu"), "apply-optimizations", "end", "lock-exit"]
    assert events.count("end") == 1


def test_train_endpoint_failure_outside_the_training_run_reports_the_error_prefix():
    events = []
    api = _training_api(events, fail="restore")
    assert api.train_hypernetwork({"steps": 1}) == ("train", "train hypernetwork error: device lost")
    assert events.count("end") == 1
    assert events[-2:] == ["end", "lock-exit"]
