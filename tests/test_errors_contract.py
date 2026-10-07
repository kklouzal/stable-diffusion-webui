from modules import errors


def test_run_reports_wrapped_exception_without_raising(capsys):
    def boom():
        raise RuntimeError("wrapped failure")

    errors.run(boom, "wrapped task")

    captured = capsys.readouterr()
    assert "wrapped task: RuntimeError" in captured.err
    assert "wrapped failure" in captured.err


def test_one_handled_exception_is_recorded_once(monkeypatch, capsys):
    monkeypatch.setattr(errors, "exception_records", [])

    try:
        raise RuntimeError("copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])")
    except RuntimeError as e:
        errors.report("loading failed", exc_info=True)
        errors.display(e, "loading")  # display() also records it again via print_error_explanation()

    records = errors.get_exceptions()
    assert len(records) == 1
    assert records[0]["exception"].startswith("copying a param")


def test_exception_records_keep_the_five_newest(monkeypatch):
    monkeypatch.setattr(errors, "exception_records", [])

    for i in range(7):
        try:
            raise ValueError(f"failure {i}")
        except ValueError:
            errors.record_exception()

    assert [record["exception"] for record in errors.get_exceptions()] == [f"failure {i}" for i in (6, 5, 4, 3, 2)]
