"""Grading logic for the invoice verification environment.

Scores an agent's ``Action`` against the ground-truth ``TaskRecord`` using:
    - Decision correctness  (0.6 base)
    - Keyword matching       (0.3 scaled proportionally)
    - Field-reference bonus  (+0.1 per referenced invoice field, capped at 1.0)
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from .models import Action, TaskRecord
from .policy import REFERENCE_FIELDS


# ---------------------------------------------------------------------------
# Tokenisation & keyword matching
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Split *text* into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _matches_keyword(reason: str, keyword: str) -> bool:
    """Return *True* if *keyword* appears (literally or token-wise) in *reason*."""
    normalized_reason = reason.lower()
    normalized_keyword = keyword.lower().strip()

    if normalized_keyword in normalized_reason:
        return True

    reason_tokens = set(_tokenize(reason))
    keyword_tokens = [token for token in _tokenize(keyword) if len(token) > 2]
    return bool(keyword_tokens) and all(token in reason_tokens for token in keyword_tokens)


def matched_keywords(reason: str, keywords: Iterable[str]) -> List[str]:
    """Return the subset of *keywords* that match *reason*."""
    return [keyword for keyword in keywords if _matches_keyword(reason, keyword)]


# ---------------------------------------------------------------------------
# Field-reference analysis
# ---------------------------------------------------------------------------

def referenced_fields(reason: str) -> Set[str]:
    """Return invoice fields (receipt / amount / category / date) cited in *reason*."""
    reason_lower = reason.lower()
    return {field for field in REFERENCE_FIELDS if field in reason_lower}


def _is_vague(reason: str) -> bool:
    """Heuristic: a reason shorter than 20 chars or with no field refs is vague."""
    if len(reason.strip()) < 20:
        return True
    return len(referenced_fields(reason)) == 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def grade(action: Action, ground_truth: TaskRecord) -> float:
    """Compute a reward ∈ [0, 1] for the agent's action.

    Breakdown:
        - 0.6  correct decision
        - 0.3  keyword coverage (proportional)
        - +0.1 per referenced invoice field (receipt, amount, category, date)
        - −0.1 penalty if reason is vague
        - Capped to [0.0, 1.0]
    """
    score = 0.0

    # ── Decision correctness ──
    if action.decision == ground_truth.decision:
        score += 0.6

    # ── Keyword coverage ──
    matches = matched_keywords(action.reason, ground_truth.keywords)
    if ground_truth.keywords:
        score += 0.3 * (len(matches) / len(ground_truth.keywords))

    # ── Field-reference bonus ──
    refs = referenced_fields(action.reason)
    score += 0.1 * len(refs)

    # ── Vagueness penalty ──
    if _is_vague(action.reason):
        score -= 0.1

    return max(0.0, min(1.0, round(score, 4)))


# ---------------------------------------------------------------------------
# Feedback builder
# ---------------------------------------------------------------------------

def build_feedback(
    action: Action,
    ground_truth: TaskRecord,
    reward: float,
) -> Dict[str, Any]:
    """Build a structured feedback dictionary for the step result ``info``.

    Keys returned:
        - ``decision_correct`` (bool)
        - ``matched_keywords`` (list[str])
        - ``referenced_fields`` (list[str])
        - ``is_vague`` (bool)
        - ``reward`` (float)
        - ``message`` (str)
    """
    decision_correct = action.decision == ground_truth.decision
    keyword_hits = matched_keywords(action.reason, ground_truth.keywords)
    refs = sorted(referenced_fields(action.reason))
    vague = _is_vague(action.reason)

    # Build human-readable message
    if decision_correct:
        message = "Correct decision."
    else:
        message = f"Incorrect decision. Expected '{ground_truth.decision}'."

    if keyword_hits:
        message += f" Reason matched policy keywords: {', '.join(keyword_hits)}."
    else:
        message += " Reason did not match any expected policy keywords."

    if refs:
        message += f" Referenced fields: {', '.join(refs)}."
    else:
        message += " No invoice fields referenced in reason."

    if vague:
        message += " Reason is too vague — consider citing specific invoice fields."

    message += f" Reward={reward:.2f}."

    return {
        "decision_correct": decision_correct,
        "matched_keywords": keyword_hits,
        "referenced_fields": refs,
        "is_vague": vague,
        "reward": reward,
        "message": message,
    }
