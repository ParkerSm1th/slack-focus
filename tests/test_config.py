from slack_priority.config import DEFAULT_THRESHOLDS, should_notify, score_to_level


def test_score_to_level_uses_thresholds():
    assert score_to_level(0.10, DEFAULT_THRESHOLDS) == "ignore"
    assert score_to_level(0.40, DEFAULT_THRESHOLDS) == "low"
    assert score_to_level(0.60, DEFAULT_THRESHOLDS) == "medium"
    assert score_to_level(0.80, DEFAULT_THRESHOLDS) == "high"
    assert score_to_level(0.95, DEFAULT_THRESHOLDS) == "critical"


def test_should_notify_defaults_to_high_plus():
    assert not should_notify("medium", "high")
    assert should_notify("high", "high")
    assert should_notify("critical", "high")
