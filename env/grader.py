"""Grading logic for the invoice verification environment.

Scores an agent's ``Action`` against the ground-truth ``TaskRecord`` using:
    - Decision correctness  (0.6 base)
    - Keyword matching       (capped)
    - Field-reference bonus  (capped)
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from .models import Action, TaskRecord
from .policy import REFERENCE_FIELDS, evaluate_invoice


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
        - 0.6  correct decision (computed dynamically from policy)
        - Explanation bonus (keywords+refs) capped at 0.4
        - −0.1 penalty if reason is vague
        - Capped to [0.0, 1.0]
    """
    # ── Policy dynamically overrides dataset label ──
    policy_result = evaluate_invoice(ground_truth.invoice)
    true_decision = policy_result["decision"]
    true_keywords = policy_result.get("violations") or ["amount", "category", "receipt", "date"]

    score = 0.0

    # ── Decision correctness ──
    if action.decision == true_decision:
        score += 0.6

    # ── Explanation Bonus ──
    matches = matched_keywords(action.reason, true_keywords)
    keyword_score = 0.0
    if true_keywords:
        keyword_score = 0.3 * (len(matches) / len(true_keywords))

    refs = referenced_fields(action.reason)
    ref_score = 0.1 * len(refs)

    # Game-proof limit: max 0.4 from explanations
    explanation_score = min(0.4, keyword_score + ref_score)
    score += explanation_score

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
    """Build a structured feedback dictionary for the step result ``info``."""
    policy_result = evaluate_invoice(ground_truth.invoice)
    true_decision = policy_result["decision"]
    true_keywords = policy_result.get("violations") or ["amount", "category", "receipt", "date"]

    decision_correct = action.decision == true_decision
    keyword_hits = matched_keywords(action.reason, true_keywords)
    refs = sorted(referenced_fields(action.reason))
    vague = _is_vague(action.reason)

    if decision_correct:
        message = "Correct decision."
    else:
        message = f"Incorrect decision. Expected '{true_decision}'."

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
