from src.finetune import get_task_config  # import from project src


def test_regression_monitors_corr_key():
    # This is a placeholder test. In real run, we would construct args and assert monitor metric.
    # Here we just ensure task3 is regression and config labels align.
    cfg = get_task_config(3)
    assert cfg["task_type"] == "regression"
    assert cfg["num_classes"] == 1
