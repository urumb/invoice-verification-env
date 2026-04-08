"""Centralized company expense policy engine.

All reimbursement rules live here so that both the grading system and
inference agent can share a single source of truth.  Nothing in this
module is environment-specific — it is pure business logic.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, FrozenSet, List, Set

# ── Policy constants ──────────────────────────────────────────────────────────

MAX_AMOUNT: float = 5000.0
"""Maximum single-invoice amount (USD) before automatic rejection."""

RECEIPT_REQUIRED: bool = True
"""Whether a receipt is required for expense approval."""

RECEIPT_WAIVER_THRESHOLD: float = 25.0
"""Amounts at or below this value may be approved without a receipt."""

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
"""Categories that are valid for company reimbursement."""

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
"""Substrings in an invoice description that signal automatic rejection."""

REFERENCE_FIELDS: List[str] = ["receipt", "amount", "category", "date"]
"""Invoice fields that a well-formed reason should reference."""


# ── Validation helpers ────────────────────────────────────────────────────────

def is_valid_category(category: str) -> bool:
    """Return *True* if the category is in the allowed set."""
    return category.lower().strip() in ALLOWED_CATEGORIES


def is_valid_amount(amount: float) -> bool:
    """Return *True* if the amount does not exceed the policy cap."""
    return 0 < amount <= MAX_AMOUNT


def is_receipt_required(amount: float) -> bool:
    """Return *True* if a receipt is required for this amount."""
    if not RECEIPT_REQUIRED:
        return False
    return amount > RECEIPT_WAIVER_THRESHOLD


def is_valid_date(date_str: str) -> bool:
    """Return *True* if the date is not in the future.

    Accepts ISO-8601 date strings (``YYYY-MM-DD``).  Malformed strings
    are treated as *invalid*.
    """
    try:
        invoice_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        return invoice_date <= date.today()
    except (ValueError, TypeError):
        return False


def has_reject_terms(description: str) -> bool:
    """Return *True* if the description contains policy-violating keywords."""
    desc_lower = description.lower()
    return any(term in desc_lower for term in REJECT_DESCRIPTION_TERMS)


def get_referenced_fields(reason: str) -> Set[str]:
    """Return the set of standard invoice fields mentioned in *reason*."""
    reason_lower = reason.lower()
    return {field for field in REFERENCE_FIELDS if field in reason_lower}


def evaluate_invoice(invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Run all policy checks on an invoice and return a diagnostic dict.

    Returns a dictionary with keys:
        - ``decision`` (``"approve"`` | ``"reject"``)
        - ``reasons`` (list of human-readable strings)
        - ``confidence`` (float 0–1)
        - ``violations`` (list of violated rule names)
    """
    description = str(invoice.get("description", "")).lower()
    category = str(invoice.get("category", "")).strip().lower()
    amount = float(invoice.get("amount", 0))
    receipt = bool(invoice.get("receipt", False))
    date_str = str(invoice.get("date", ""))

    violations: List[str] = []
    reasons: List[str] = []

    # Rule 1 — receipt
    if not receipt and is_receipt_required(amount):
        violations.append("missing_receipt")
        reasons.append(
            f"Rejected: receipt is missing and amount ${amount:.2f} exceeds "
            f"the ${RECEIPT_WAIVER_THRESHOLD:.2f} waiver threshold."
        )

    # Rule 2 — amount cap
    if not is_valid_amount(amount):
        violations.append("excessive_amount")
        reasons.append(
            f"Rejected: amount ${amount:.2f} exceeds the policy cap of "
            f"${MAX_AMOUNT:.2f}."
        )

    # Rule 3 — category
    if not is_valid_category(category):
        violations.append("invalid_category")
        reasons.append(
            f"Rejected: category '{category}' is not in the list of "
            f"allowed categories."
        )

    # Rule 4 — future date
    if not is_valid_date(date_str):
        violations.append("invalid_date")
        reasons.append(
            f"Rejected: date '{date_str}' is invalid or in the future."
        )

    # Rule 5 — description red flags
    if has_reject_terms(description):
        violations.append("description_red_flag")
        reasons.append(
            "Rejected: description contains terms that violate expense policy."
        )

    if violations:
        confidence = 0.9 if len(violations) >= 2 else 0.7
        return {
            "decision": "reject",
            "reasons": reasons,
            "confidence": confidence,
            "violations": violations,
        }

    # All checks passed
    return {
        "decision": "approve",
        "reasons": [
            f"Approved: amount ${amount:.2f} is within policy, "
            f"category '{category}' is allowed, receipt is present, "
            f"and date '{date_str}' is valid."
        ],
        "confidence": 0.9,
        "violations": [],
    }
