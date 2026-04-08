"""Centralized company expense policy engine."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, FrozenSet, List, Optional, Set


MAX_AMOUNT: float = 2000.0
RECEIPT_REQUIRED: bool = True

ALLOWED_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "office_supplies",
        "office_equipment",
        "electronics",
        "software",
        "training",
        "travel",
        "transportation",
        "infrastructure",
        "legal",
        "accommodation",
        "health",
        "rent",
        "meals",
    }
)

REJECT_DESCRIPTION_TERMS: List[str] = [
    "personal",
    "birthday",
    "gaming",
    "grocery",
    "household",
    "luxury",
    "alcohol",
    "first-class",
    "first class",
    "spa",
    "video game",
]

REFERENCE_FIELDS: List[str] = ["receipt", "amount", "category", "date"]


def normalize_category(category: Any) -> str:
    return str(category or "").strip().lower()


def parse_amount(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_receipt(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def is_valid_category(category: str) -> bool:
    return normalize_category(category) in ALLOWED_CATEGORIES


def is_valid_amount(amount: Optional[float]) -> bool:
    return amount is not None and 0 < amount <= MAX_AMOUNT


def is_valid_date(date_str: str) -> bool:
    try:
        invoice_date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
        return invoice_date <= date.today()
    except (TypeError, ValueError):
        return False


def has_reject_terms(description: str) -> bool:
    desc_lower = str(description or "").lower()
    return any(term in desc_lower for term in REJECT_DESCRIPTION_TERMS)


def get_referenced_fields(reason: str) -> Set[str]:
    reason_lower = str(reason or "").lower()
    return {field for field in REFERENCE_FIELDS if field in reason_lower}


def _format_amount(amount: Optional[float]) -> str:
    if amount is None:
        return "unknown"
    return f"${amount:.2f}"


def evaluate_invoice(invoice: Dict[str, Any]) -> Dict[str, Any]:
    description = str(invoice.get("description", "")).strip()
    description_lower = description.lower()
    category = normalize_category(invoice.get("category", ""))
    amount = parse_amount(invoice.get("amount"))
    receipt = parse_receipt(invoice.get("receipt", False))
    date_str = str(invoice.get("date", "")).strip()

    violations: List[str] = []
    reasons: List[str] = []
    expected_reasoning: List[str] = []

    if amount is None or amount <= 0:
        violations.append("invalid_amount")
        reasons.append("Rejected: amount is missing, non-numeric, or not greater than zero.")
        expected_reasoning.append("amount is invalid")
    elif amount > MAX_AMOUNT:
        violations.append("excessive_amount")
        reasons.append(
            f"Rejected: amount {_format_amount(amount)} exceeds the policy limit of {_format_amount(MAX_AMOUNT)}."
        )
        expected_reasoning.append(
            f"amount {_format_amount(amount)} exceeds the policy limit of {_format_amount(MAX_AMOUNT)}"
        )

    if RECEIPT_REQUIRED and not receipt:
        violations.append("missing_receipt")
        reasons.append("Rejected: receipt is required for every reimbursable invoice.")
        expected_reasoning.append("receipt is missing")

    if not is_valid_category(category):
        violations.append("invalid_category")
        reasons.append(
            f"Rejected: category '{category or 'unknown'}' is not an allowed reimbursement category."
        )
        expected_reasoning.append(f"category {category or 'unknown'} is not allowed")

    if not is_valid_date(date_str):
        violations.append("invalid_date")
        reasons.append(f"Rejected: date '{date_str or 'unknown'}' is invalid or in the future.")
        expected_reasoning.append(f"date {date_str or 'unknown'} is invalid")

    if has_reject_terms(description):
        matched_terms = [term for term in REJECT_DESCRIPTION_TERMS if term in description_lower]
        descriptor = matched_terms[0] if matched_terms else "policy-violating term"
        violations.append("description_red_flag")
        reasons.append(
            f"Rejected: description contains the red-flag term '{descriptor}', which violates expense policy."
        )
        expected_reasoning.append(f"description contains disallowed term {descriptor}")

    if violations:
        confidence = 0.95 if len(violations) >= 2 else 0.88
        return {
            "decision": "reject",
            "reasons": reasons,
            "confidence": confidence,
            "violations": violations,
            "expected_reasoning": expected_reasoning,
        }

    approval_reasons = [
        f"Approved: amount {_format_amount(amount)} is within the {_format_amount(MAX_AMOUNT)} policy limit.",
        f"Approved: category '{category}' is in the allowed reimbursement list.",
        "Approved: receipt is present.",
        f"Approved: date '{date_str}' is valid and not in the future.",
    ]
    approval_reasoning = [
        f"amount {_format_amount(amount)} is within the {_format_amount(MAX_AMOUNT)} policy limit",
        f"category {category} is allowed",
        "receipt is present",
        f"date {date_str} is valid",
    ]

    return {
        "decision": "approve",
        "reasons": approval_reasons,
        "confidence": 0.92,
        "violations": [],
        "expected_reasoning": approval_reasoning,
    }
