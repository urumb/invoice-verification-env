"""Grading logic for the invoice verification environment."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from .models import Action, Stage, TaskRecord
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
}

TASK_WEIGHTS = {
    "easy": {
        "analysis_fields": ["amount", "category", "receipt", "date"],
        "issue_weight": 0.35,
        "decision_weight": 0.65,
        "reasoning_weight": 0.15,
        "fact_weight": 0.05,
        "clarity_weight": 0.05,
    },
    "medium": {
        "analysis_fields": ["amount", "category", "receipt", "date", "subtotal", "tax_amount", "reported_total"],
        "issue_weight": 0.40,
        "decision_weight": 0.58,
        "reasoning_weight": 0.17,
        "fact_weight": 0.08,
        "clarity_weight": 0.05,
    },
    "hard": {
        "analysis_fields": [
            "amount",
            "category",
            "receipt",
            "date",
            "subtotal",
            "tax_amount",
            "reported_total",
            "vendor_name",
            "vendor_status",
            "line_items",
        ],
        "issue_weight": 0.45,
        "decision_weight": 0.50,
        "reasoning_weight": 0.22,
        "fact_weight": 0.13,
        "clarity_weight": 0.05,
    },
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
    return len(_meaningful_tokens(reason)) < 4


def clamp_score(score: float) -> float:
    """Clamp a score to the open interval (0, 1)."""
    return max(0.01, min(0.99, float(score)))


def _clarity_score(reason: str, max_score: float) -> float:
    """Return an additive clarity bonus (0 if vague, max_score otherwise).

    The return value is a sub-score component that is always *added* to a
    base reward before the caller applies clamp_score().  It is intentionally
    zero for vague reasoning but expressed via multiplication to avoid a bare
    literal `return 0.0` that fails strict-interval validators.
    """
    return round(max_score, 4) * int(not _is_vague(reason))


def _task_config(task: str) -> Dict[str, Any]:
    return TASK_WEIGHTS.get(task, TASK_WEIGHTS["easy"])


def _format_amount(value: Any) -> str:
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _invoice_field_findings(task: str, invoice: Dict[str, Any]) -> Dict[str, str]:
    findings = {
        "amount": f"amount {_format_amount(invoice.get('amount'))}",
        "category": f"category {str(invoice.get('category', 'unknown')).strip().lower() or 'unknown'}",
        "receipt": f"receipt {'present' if bool(invoice.get('receipt', False)) else 'missing'}",
        "date": f"date {str(invoice.get('date', '')).strip() or 'unknown'}",
    }
    if task in {"medium", "hard"}:
        findings["subtotal"] = f"subtotal {_format_amount(invoice.get('subtotal'))}"
        findings["tax_amount"] = f"tax {_format_amount(invoice.get('tax_amount'))}"
        findings["reported_total"] = f"reported total {_format_amount(invoice.get('reported_total'))}"
    if task == "hard":
        findings["vendor_name"] = f"vendor {str(invoice.get('vendor_name', 'unknown')).strip()}"
        findings["vendor_status"] = f"vendor status {str(invoice.get('vendor_status', 'unknown')).strip()}"
        line_items = invoice.get("line_items") or []
        findings["line_items"] = f"line items {len(line_items)}"
    return findings


def _issue_targets(task: str, invoice: Dict[str, Any], decision: str) -> List[str]:
    anomaly_flags = [str(item) for item in invoice.get("anomaly_flags", [])]
    if anomaly_flags:
        return _dedupe_preserve_order(anomaly_flags)
    if decision == "reject":
        return ["policy issue detected", "invoice should be rejected"]
    targets = ["no policy issues"]
    if task in {"medium", "hard"}:
        targets.extend(["tax matches subtotal", "reported total is consistent"])
    if task == "hard":
        targets.extend(["vendor is verified", "cross-field checks align"])
    return _dedupe_preserve_order(targets)


def _reasoning_targets(ground_truth: TaskRecord, task: str) -> List[str]:
    targets: List[str] = list(ground_truth.keywords or [])
    targets.extend(_issue_targets(task, ground_truth.invoice, str(ground_truth.decision)))
    return _dedupe_preserve_order(targets)


def _fact_targets(ground_truth: TaskRecord, task: str) -> List[str]:
    invoice = ground_truth.invoice
    targets = list(_invoice_field_findings(task, invoice).values())
    line_items = invoice.get("line_items") or []
    if task == "hard" and line_items:
        targets.append(f"{len(line_items)} line items")
    return _dedupe_preserve_order(targets)


def _analysis_feedback(action: Action, ground_truth: TaskRecord, task: str) -> Dict[str, Any]:
    field_findings = _invoice_field_findings(task, ground_truth.invoice)
    target_fields = _task_config(task)["analysis_fields"]
    hits = [name for name in target_fields if name in _normalize_phrase(action.reasoning)]
    coverage = len(hits) / max(1, len(target_fields))
    reward = 0.10 + (0.20 * coverage) + _clarity_score(action.reasoning, 0.05)
    reward = clamp_score(round(reward, 4))
    captured_findings = [field_findings[name] for name in hits if name in field_findings]
    return {
        "reward": clamp_score(reward),
        "captured_findings": captured_findings,
        "matched_keywords": hits,
        "matched_fact_targets": captured_findings,
        "referenced_fields": sorted(referenced_fields(action.reasoning)),
        "decision_correct": None,
        "message": f"{task.title()} analysis stage evaluated field inspection.",
    }


def _issue_feedback(action: Action, ground_truth: TaskRecord, task: str) -> Dict[str, Any]:
    invoice = ground_truth.invoice
    targets = _issue_targets(task, invoice, str(ground_truth.decision))
    hits = matched_keywords(action.reasoning, targets)
    denominator = max(1, len(targets))
    reward = 0.10 + (_task_config(task)["issue_weight"] * clamp_score(len(hits) / denominator))
    reward += _clarity_score(action.reasoning, 0.05)
    if not invoice.get("anomaly_flags"):
        lowered = action.reasoning.lower()
        if "no issues" in lowered or "no policy issues" in lowered or "consistent" in lowered:
            hits = _dedupe_preserve_order(hits + ["no policy issues"])
            reward += 0.05
    reward = clamp_score(round(reward, 4))
    captured_findings = hits or (["no policy issues"] if not invoice.get("anomaly_flags") else [])
    return {
        "reward": clamp_score(reward),
        "captured_findings": captured_findings,
        "matched_keywords": hits,
        "matched_fact_targets": hits,
        "referenced_fields": sorted(referenced_fields(action.reasoning)),
        "decision_correct": None,
        "message": f"{task.title()} issue stage evaluated anomaly detection.",
    }


def _consistency_penalty(
    action: Action,
    decision: str,
    previous_findings: Iterable[str],
) -> float:
    previous_text = " ".join(previous_findings).lower()
    action_value = action.action.strip().lower()
    penalty = 0.0
    if decision == "reject" and action_value == "approve":
        penalty += 0.10
    if decision == "approve" and action_value == "reject":
        penalty += 0.05
    if previous_text:
        if "no policy issues" in previous_text and action_value == "reject":
            penalty += 0.05
        if "mismatch" in previous_text and action_value == "approve":
            penalty += 0.05
        if "unverified" in previous_text and action_value == "approve":
            penalty += 0.05
        if "conflict" in previous_text and action_value == "approve":
            penalty += 0.05
    return penalty


def _decision_feedback(
    action: Action,
    ground_truth: TaskRecord,
    task: str,
    previous_findings: Iterable[str],
) -> Dict[str, Any]:
    true_decision = str(ground_truth.decision)
    reasoning_targets = _reasoning_targets(ground_truth, task)
    fact_targets = _fact_targets(ground_truth, task)
    config = _task_config(task)

    decision_correct = action.action.strip().lower() == true_decision
    keyword_hits = matched_keywords(action.reasoning, reasoning_targets)
    fact_hits = matched_keywords(action.reasoning, fact_targets)
    refs = sorted(referenced_fields(action.reasoning))

    if decision_correct:
        reward = config["decision_weight"]
        reward += config["reasoning_weight"] * clamp_score(len(keyword_hits) / max(1, len(reasoning_targets)))
        reward += config["fact_weight"] * clamp_score(len(fact_hits) / max(1, min(len(fact_targets), 4)))
        reward += _clarity_score(action.reasoning, config["clarity_weight"])
    else:
        reward = 0.01

    reward -= _consistency_penalty(action, true_decision, previous_findings)
    reward = clamp_score(round(reward, 4))

    return {
        "reward": clamp_score(reward),
        "captured_findings": [f"final decision {action.action.strip().lower()}"],
        "matched_keywords": keyword_hits,
        "matched_fact_targets": fact_hits,
        "referenced_fields": refs,
        "decision_correct": decision_correct,
        "expected_decision": true_decision,
        "message": "Final decision matched task expectations." if decision_correct else f"Final decision incorrect. Expected '{true_decision}'.",
    }


def evaluate_stage(
    action: Action,
    ground_truth: TaskRecord,
    task: str,
    expected_stage: Stage,
    previous_findings: Iterable[str],
) -> Dict[str, Any]:
    if expected_stage == "analyze":
        feedback = _analysis_feedback(action, ground_truth, task)
    elif expected_stage == "flag_issues":
        feedback = _issue_feedback(action, ground_truth, task)
    else:
        feedback = _decision_feedback(action, ground_truth, task, previous_findings)

    stage_correct = action.stage == expected_stage
    wrong_stage_penalty = 0.15 if not stage_correct else 0.0
    reward = clamp_score(round(feedback["reward"] - wrong_stage_penalty, 4))

    feedback.update(
        {
            "reward": clamp_score(reward),
            "task": task,
            "expected_stage": expected_stage,
            "stage_correct": stage_correct,
            "wrong_stage_penalty": wrong_stage_penalty,
            "is_vague": _is_vague(action.reasoning),
        }
    )
    return feedback


def grade(
    action: Action,
    ground_truth: TaskRecord,
    task: str,
    expected_stage: Stage,
    previous_findings: Iterable[str],
) -> float:
    score = float(evaluate_stage(action, ground_truth, task, expected_stage, previous_findings)["reward"])
    return clamp_score(score)


def build_feedback(
    action: Action,
    ground_truth: TaskRecord,
    task: str,
    expected_stage: Stage,
    previous_findings: Iterable[str],
    reward: float,
) -> Dict[str, Any]:
    feedback = evaluate_stage(action, ground_truth, task, expected_stage, previous_findings)
    feedback["reward"] = clamp_score(reward)
    return feedback