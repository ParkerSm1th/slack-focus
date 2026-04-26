from slack_priority.text import clean_text, tokenize


def test_clean_text_masks_slack_and_external_tokens():
    cleaned = clean_text("Hey <@U123>, see <#C456|incidents> https://example.com `secret` a@b.com")
    assert "user" in cleaned
    assert "channel" in cleaned
    assert "url" in cleaned
    assert "code" in cleaned
    assert "email" in cleaned
    assert "u123" not in cleaned
    assert "example.com" not in cleaned


def test_tokenize_lowercases_words():
    assert tokenize("PROD is Down") == ["prod", "is", "down"]
