import pytest

from slack_priority.diagnostics import (
    DiagnosticExample,
    format_overfit_report,
    run_overfit_check,
)


def _examples():
    urgent = [
        "prod outage checkout down",
        "security incident credentials exposed",
        "customer escalation blocking launch",
        "database errors impacting customers",
        "auth service critical failure",
        "payment system down urgent",
        "release blocked needs approval",
        "major incident all users affected",
        "vpn outage remote team blocked",
        "high priority customer cannot login",
        "urgent deadline deployment blocked",
        "data loss incident immediate response",
    ]
    routine = [
        "weekly status update",
        "team lunch tomorrow",
        "minor typo in docs",
        "newsletter for later",
        "routine documentation question",
        "pricing question can wait",
        "nice to have roadmap idea",
        "general product inquiry",
        "fyi office snacks arrived",
        "calendar reminder for next week",
        "low priority cleanup task",
        "casual team update no action",
    ]
    return [
        DiagnosticExample(text=text, score=0.98, source="local", weight=5.0)
        for text in urgent
    ] + [
        DiagnosticExample(text=text, score=0.0, source="local", weight=5.0)
        for text in routine
    ]


def test_overfit_check_returns_report_without_saving_model():
    report = run_overfit_check(
        _examples(),
        epochs=1,
        batch_size=4,
        max_vocab=64,
        min_freq=1,
        shuffle_baseline=True,
    )

    assert report["examples"]["total"] == 24
    assert report["examples"]["train"] > 0
    assert report["examples"]["validation"] > 0
    assert report["settings"]["vocab_size"] > 0
    assert 0.0 <= report["train"]["mae"] <= 1.0
    assert 0.0 <= report["validation"]["mae"] <= 1.0
    assert "constant_validation_mae" in report["baselines"]
    assert "Overfit check" in format_overfit_report(report)


def test_overfit_check_requires_enough_examples():
    with pytest.raises(ValueError, match="Need at least 20"):
        run_overfit_check(_examples()[:5], epochs=1)
