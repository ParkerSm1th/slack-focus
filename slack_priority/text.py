from __future__ import annotations

import re
from collections import Counter

import numpy as np


TOKEN_RE = re.compile(r"[a-z0-9_']+")
URL_RE = re.compile(r"https?://\S+|www\.\S+|<https?://[^>]+>")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
SLACK_USER_RE = re.compile(r"<@[UW][A-Z0-9]+(?:\|[^>]+)?>")
SLACK_CHANNEL_RE = re.compile(r"<#[C][A-Z0-9]+(?:\|[^>]+)?>")
CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`]+`")


def clean_text(text: str) -> str:
    text = CODE_BLOCK_RE.sub(" CODE_BLOCK ", text)
    text = INLINE_CODE_RE.sub(" CODE ", text)
    text = URL_RE.sub(" URL ", text)
    text = EMAIL_RE.sub(" EMAIL ", text)
    text = SLACK_USER_RE.sub(" USER ", text)
    text = SLACK_CHANNEL_RE.sub(" CHANNEL ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean_text(text))


def build_vocab(texts: list[str], *, max_tokens: int = 4096, min_freq: int = 2) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))

    most_common = [
        token
        for token, count in counts.most_common()
        if count >= min_freq and len(token) > 1
    ][:max_tokens]
    return {token: idx for idx, token in enumerate(most_common)}


def vectorize(texts: list[str], vocab: dict[str, int]) -> np.ndarray:
    features = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = tokenize(text)
        if not tokens:
            continue

        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                features[row, idx] += 1.0

        norm = np.sqrt(max(float(features[row].sum()), 1.0))
        features[row] /= norm

    return features
