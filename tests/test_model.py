from slack_priority.storage import Store


def test_train_save_and_score_tiny_model(tmp_path, monkeypatch):
    import slack_priority.model as model_module

    monkeypatch.setattr(model_module, "MODEL_PATH", tmp_path / "model.safetensors")
    monkeypatch.setattr(model_module, "VOCAB_PATH", tmp_path / "vocab.json")
    monkeypatch.setattr(model_module, "METRICS_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(
        model_module,
        "fetch_public_examples",
        lambda max_rows: [
            ("prod checkout outage customers cannot pay", 0.85),
            ("security incident data loss immediate response", 0.98),
            ("routine documentation question for later", 0.15),
            ("weekly status update no action required", 0.15),
            ("performance issue blocking deployment today", 0.85),
            ("feature information request", 0.50),
            ("billing clarification can wait", 0.15),
            ("major incident all users down", 0.98),
            ("minor typo in docs", 0.15),
            ("customer escalation urgent deadline", 0.85),
            ("nice to have idea for roadmap", 0.15),
            ("vpn outage affecting remote workers", 0.85),
            ("marketing pricing question", 0.50),
            ("password reset question", 0.50),
            ("critical auth service down", 0.98),
            ("team lunch tomorrow", 0.15),
            ("blocked on release approval today", 0.85),
            ("newsletter update", 0.15),
            ("database errors impacting checkout", 0.98),
            ("general product inquiry", 0.50),
        ],
    )

    store = Store(tmp_path / "test.sqlite3")
    try:
        store.upsert_message(message_id="m1", text="deploy is blocked and deadline is today")
        store.set_label("m1", "high")
        result = model_module.train_model(
            store=store,
            public_limit=20,
            epochs=1,
            batch_size=4,
            max_vocab=64,
            min_freq=1,
        )
    finally:
        store.close()

    scorer = model_module.UrgencyScorer(
        model_path=tmp_path / "model.safetensors",
        vocab_path=tmp_path / "vocab.json",
    )
    score, level = scorer.score_level("critical outage customers cannot pay")

    assert result.metrics["vocab_size"] > 0
    assert scorer.ready
    assert scorer._predict is not None
    assert 0.0 <= score <= 1.0
    assert level in {"ignore", "low", "medium", "high", "critical"}
