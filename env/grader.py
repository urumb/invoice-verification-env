"""Grading logic for the invoice verification environment."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from .models import Action, TaskRecord
from .policy import REFERENCE_FIELDS, evaluate_invoice


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "because",
    "by",
    "for",
    "from",
    "in",
    "invoice",
    "is",
    "of",
    "on",
    "policy",
    "that",
    "the",
    "this",
    "to",
    "with",
    "amount",
    "category",
    "date",
    "receipt",
}


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def _normalize_phrase(text: str) -> str:
    return " ".join(_tokenize(text))


def _meaningful_tokens(text: str) -> List[str]:
    return [
        token
        for token in _tokenize(text)
        if token not in STOPWORDS and len(token) > 1
    ]


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        normalized = _normalize_phrase(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(str(value))
    return result


def _matches_concept(reason: str, concept: str) -> bool:
    normalized_reason = _normalize_phrase(reason)
    normalized_concept = _normalize_phrase(concept)
    if not normalized_reason or not normalized_concept:
        return False

    if normalized_concept in normalized_reason:
        return True

    reason_tokens = set(_meaningful_tokens(reason))
    concept_tokens = _meaningful_tokens(concept)
    if not concept_tokens:
        return False

    return all(token in reason_tokens for token in concept_tokens)


def matched_keywords(reason: str, keywords: Iterable[str]) -> List[str]:
    return [keyword for keyword in keywords if _matches_concept(reason, keyword)]


def referenced_fields(reason: str) -> Set[str]:
    reason_lower = (reason or "").lower()
    return {field for field in REFERENCE_FIELDS if field in reason_lower}


def _is_vague(reason: str) -> bool:
    meaningful = _meaningful_tokens(reason)
    return len(meaningful) < 4


def _reasoning_targets(ground_truth: TaskRecord, policy_result: Dict[str, Any]) -> List[str]:
    targets: List[str] = []
    targets.extend(ground_truth.keywords or [])
    targets.extend(policy_result.get("expected_reasoning") or [])
    return _dedupe_preserve_order(targets)


def _fact_targets(ground_truth: TaskRecord, policy_result: Dict[str, Any]) -> List[str]:
    invoice = ground_truth.invoice
    targets: List[str] = list(policy_result.get("expected_reasoning") or [])

    category = str(invoice.get("category", "")).strip().lower()
    if category:
        targets.append(category.replace("_", " "))

    amount_value = invoice.get("amount")
    try:
        amount = float(amount_value)
    except (TypeError, ValueError):
        amount = None
    if amount is not None:
        targets.append(f"${amount:.2f}")
        targets.append(f"{amount:g}")

    receipt = bool(invoice.get("receipt", False))
    targets.append("receipt is present" if receipt else "receipt is missing")

    date_str = str(invoice.get("date", "")).strip()
    if date_str:
        targets.append(date_str)

    return _dedupe_preserve_order(targets)


def _relevance_score(reason: str, targets: List[str]) -> float:
    if not reason.strip() or not targets:
        return 0.0
    hits = matched_keywords(reason, targets)
    return round(0.25 * (len(hits) / len(targets)), 4)


def _specificity_score(reason: str, fact_targets: List[str]) -> float:
    if not reason.strip() or not fact_targets:
        return 0.0
    hits = matched_keywords(reason, fact_targets)
    if not hits:
        return 0.0
    denominator = max(1, min(len(fact_targets), 2))
    return round(0.10 * min(1.0, len(hits) / denominator), 4)


def _clarity_score(reason: str) -> float:
    if _is_vague(reason):
        return 0.0
    if len(_tokenize(reason)) < 6:
        return 0.0
    return 0.05


def grade(action: Action, ground_truth: TaskRecord) -> float:
    policy_result = evaluate_invoice(ground_truth.invoice)
    true_decision = policy_result["decision"]
    if action.decision != true_decision:
        return 0.0

    targets = _reasoning_targets(ground_truth, policy_result)
    fact_targets = _fact_targets(ground_truth, policy_result)

    decision_score = 0.55
    relevance_score = _relevance_score(action.reason, targets)
    specificity_score = _specificity_score(action.reason, fact_targets)
    clarity_score = _clarity_score(action.reason)

    total = decision_score + relevance_score + specificity_score + clarity_score
    return max(0.0, min(0.95, round(total, 4)))


def build_feedback(
    action: Action,
    ground_truth: TaskRecord,
    reward: float,
) -> Dict[str, Any]:
    policy_result = evaluate_invoice(ground_truth.invoice)
    true_decision = policy_result["decision"]
    reasoning_targets = _reasoning_targets(ground_truth, policy_result)
    fact_targets = _fact_targets(ground_truth, policy_result)

    decision_correct = action.decision == true_decision
    keyword_hits = matched_keywords(action.reason, reasoning_targets)
    fact_hits = matched_keywords(action.reason, fact_targets)
    refs = sorted(referenced_fields(action.reason))
    vague = _is_vague(action.reason)
    relevance_score = _relevance_score(action.reason, reasoning_targets) if decision_correct else 0.0
    specificity_score = _specificity_score(action.reason, fact_targets) if decision_correct else 0.0
    clarity_score = _clarity_score(action.reason) if decision_correct else 0.0

    if decision_correct:
        message = "Correct decision."
    else:
        message = f"Incorrect decision. Expected '{true_decision}'."

    if keyword_hits:
        message += f" Reason matched relevant evidence: {', '.join(keyword_hits)}."
    else:
        message += " Reason did not match the expected policy evidence."

    if fact_hits:
        message += f" Invoice-specific facts referenced: {', '.join(fact_hits)}."
    else:
        message += " Reason did not reference invoice-specific facts."

    if refs:
        message += f" Referenced fields: {', '.join(refs)}."
    else:
        message += " No invoice fields referenced in reason."

    if vague:
        message += " Reason is vague."

    message += (
        f" Relevance score={relevance_score:.2f}."
        f" Specificity score={specificity_score:.2f}."
        f" Clarity score={clarity_score:.2f}."
        f" Reward={reward:.2f}."
    )

    return {
        "decision_correct": decision_correct,
        "expected_decision": true_decision,
        "matched_keywords": keyword_hits,
        "matched_fact_targets": fact_hits,
        "referenced_fields": refs,
        "is_vague": vague,
        "relevance_score": relevance_score,
        "specificity_score": specificity_score,
        "clarity_score": clarity_score,
        "reward": reward,
        "message": message,
    }
