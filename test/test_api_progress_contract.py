def test_progress_response_declares_current_task_returned_by_api(initialize):
    # progressapi returns current_task (test_postprocessing_api_defaults.py::test_api_progress_reports_live_current_task_reference);
    # the response model drops undeclared keys, so it must declare the field.
    from modules.api import models

    payload = {"progress": 0.5, "eta_relative": 12.0, "state": {}, "current_task": "task(txt2img-LIVE)"}

    assert models.ProgressResponse.model_validate(payload).model_dump()["current_task"] == "task(txt2img-LIVE)"
    assert models.ProgressResponse.model_validate({**payload, "current_task": None}).current_task is None
    assert models.ProgressResponse.model_validate({key: payload[key] for key in ("progress", "eta_relative", "state")}).current_task is None
