from pathlib import Path


def test_progress_response_declares_current_task_returned_by_api():
    api_source = Path("modules/api/api.py").read_text()
    model_source = Path("modules/api/models.py").read_text()

    assert "current_task=progress_module.current_task" in api_source
    assert "current_task: Optional[str]" in model_source
