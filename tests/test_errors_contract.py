from modules import errors


def test_run_reports_wrapped_exception_without_raising(capsys):
    def boom():
        raise RuntimeError("wrapped failure")

    errors.run(boom, "wrapped task")

    captured = capsys.readouterr()
    assert "wrapped task: RuntimeError" in captured.err
    assert "wrapped failure" in captured.err
