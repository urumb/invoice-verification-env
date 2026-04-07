from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .models import Action, TaskRecord


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _matches_keyword(reason: str, keyword: str) -> bool:
    normalized_reason = reason.lower()
    normalized_keyword = keyword.lower().strip()

    if normalized_keyword in normalized_reason:
        return True

    reason_tokens = set(_tokenize(reason))
    keyword_tokens = [token for token in _tokenize(keyword) if len(token) > 2]
    return bool(keyword_tokens) and all(token in reason_tokens for token in keyword_tokens)


def matched_keywords(reason: str, keywords: Iterable[str]) -> List[str]:
    return [keyword for keyword in keywords if _matches_keyword(reason, keyword)]


def grade(action: Action, ground_truth: TaskRecord) -> float:
    score = 0.0

    if action.decision == ground_truth.decision:
        score += 0.7

    matches = matched_keywords(action.reason, ground_truth.keywords)
    if ground_truth.keywords:
        score += 0.3 * (len(matches) / len(ground_truth.keywords))

    return max(0.0, min(1.0, round(score, 4)))


def build_feedback(action: Action, ground_truth: TaskRecord, reward: float) -> Dict[str, Any]:
    decision_correct = action.decision == ground_truth.decision
    keyword_hits = matched_keywords(action.reason, ground_truth.keywords)

    if decision_correct:
        message = "Correct decision."
    else:
        message = f"Incorrect decision. Expected '{ground_truth.decision}'."

    if keyword_hits:
        message += f" Reason matched policy keywords: {', '.join(keyword_hits)}."
    else:
        message += " Reason did not match any expected policy keywords."

    message += f" Reward={reward:.2f}."

    return {
        "decision_correct": decision_correct,
        "matched_keywords": keyword_hits,
        "reward": reward,
        "message": message,
    }
