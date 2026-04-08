"""Grading logic for the invoice verification environment."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from .models import Action, TaskRecord
from .policy import REFERENCE_FIELDS, evaluate_invoice


GENERIC_FIELD_TOKENS = {"amount", "category", "receipt", "date"}
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Split *text* into lowercase alphanumeric tokens."""
    return TOKEN_PATTERN.findall((text or "").lower())


def _normalize_phrase(text: str) -> str:
    """Collapse *text* into a normalized space-delimited token string."""
    return " ".join(_tokenize(text))


def _extract_expected_keywords(violations: Iterable[str]) -> List[str]:
    """Convert policy violations into de-duplicated explanation keywords."""
    expected: List[str] = []
    seen: Set[str] = set()

    for violation in violations:
        keyword = _normalize_phrase(str(violation).replace("_", " "))
        if not keyword:
            continue

        meaningful_tokens = [
            token
            for token in _tokenize(keyword)
            if token not in GENERIC_FIELD_TOKENS and len(token) > 2
        ]
        if not meaningful_tokens:
            continue

        if keyword not in seen:
            seen.add(keyword)
            expected.append(keyword)

    return expected


def _matches_keyword(reason: str, keyword: str) -> bool:
    """Return *True* when *reason* meaningfully matches *keyword*."""
    normalized_reason = _normalize_phrase(reason)
    normalized_keyword = _normalize_phrase(keyword)
    if not normalized_reason or not normalized_keyword:
        return False

    if normalized_keyword in normalized_reason:
        return True

    reason_tokens = set(_tokenize(reason))
    keyword_tokens = _tokenize(keyword)
    meaningful_tokens = [
        token for token in keyword_tokens if token not in GENERIC_FIELD_TOKENS and len(token) > 2
    ]
    if not meaningful_tokens:
        return False

    if len(meaningful_tokens) == 1 and len(keyword_tokens) > 1:
        return all(token in reason_tokens for token in keyword_tokens)

    return all(token in reason_tokens for token in meaningful_tokens)


def matched_keywords(reason: str, keywords: Iterable[str]) -> List[str]:
    """Return the subset of *keywords* that match *reason*."""
    return [keyword for keyword in keywords if _matches_keyword(reason, keyword)]


def referenced_fields(reason: str) -> Set[str]:
    """Return invoice fields cited in *reason*."""
    reason_lower = (reason or "").lower()
    return {field for field in REFERENCE_FIELDS if field in reason_lower}


def _is_vague(reason: str) -> bool:
    """Heuristic flag for very short or content-free reasoning."""
    tokens = _tokenize(reason)
    if len(tokens) < 3:
        return True
    return len(referenced_fields(reason)) == 0 and len(tokens) < 5


def _reasoning_bonus(reason: str) -> float:
    """Award a small bonus for non-empty reasoning without over-rewarding trivial text."""
    token_count = len(_tokenize(reason))
    if token_count == 0:
        return 0.0
    return min(0.1, round(0.02 * token_count, 4))


def _keyword_score(reason: str, expected_keywords: List[str]) -> float:
    """Score explanation quality using only meaningful policy keywords."""
    if not expected_keywords:
        return 0.0

    matches = matched_keywords(reason, expected_keywords)
    denominator = max(len(expected_keywords), 2)
    return min(0.3, round(0.3 * (len(matches) / denominator), 4))


def grade(action: Action, ground_truth: TaskRecord) -> float:
    """Compute a reward in ``[0, 1]`` for the agent's action."""
    policy_result = evaluate_invoice(ground_truth.invoice)
    true_decision = policy_result["decision"]
    expected_keywords = _extract_expected_keywords(policy_result.get("violations") or [])

    base = 0.6 if action.decision == true_decision else 0.0
    explanation_score = _keyword_score(action.reason, expected_keywords)
    reasoning_bonus = _reasoning_bonus(action.reason)

    total = base + explanation_score + reasoning_bonus
    return max(0.0, min(1.0, round(total, 4)))


def build_feedback(
    action: Action,
    ground_truth: TaskRecord,
    reward: float,
) -> Dict[str, Any]:
    """Build a structured feedback dictionary for the step result ``info``."""
    policy_result = evaluate_invoice(ground_truth.invoice)
    true_decision = policy_result["decision"]
    expected_keywords = _extract_expected_keywords(policy_result.get("violations") or [])

    decision_correct = action.decision == true_decision
    keyword_hits = matched_keywords(action.reason, expected_keywords)
    refs = sorted(referenced_fields(action.reason))
    vague = _is_vague(action.reason)
    explanation_score = _keyword_score(action.reason, expected_keywords)
    reasoning_bonus = _reasoning_bonus(action.reason)

    if decision_correct:
        message = "Correct decision."
    else:
        message = f"Incorrect decision. Expected '{true_decision}'."

    if expected_keywords:
        if keyword_hits:
            message += f" Reason matched policy keywords: {', '.join(keyword_hits)}."
        else:
            message += " Reason did not match any expected policy keywords."
    else:
        message += " No explanation keywords were expected for this approval case."

    if refs:
        message += f" Referenced fields: {', '.join(refs)}."
    else:
        message += " No invoice fields referenced in reason."

    if vague:
        message += " Reason is vague."

    message += (
        f" Explanation score={explanation_score:.2f}."
        f" Reasoning bonus={reasoning_bonus:.2f}."
        f" Reward={reward:.2f}."
    )

    return {
        "decision_correct": decision_correct,
        "matched_keywords": keyword_hits,
        "referenced_fields": refs,
        "is_vague": vague,
        "reward": reward,
        "message": message,
    }
