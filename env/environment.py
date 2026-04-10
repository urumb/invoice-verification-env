from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, cast

from .grader import build_feedback, grade
from .models import Action, Difficulty, Observation, State, StepResult, TaskRecord
from .tasks import get_random_task


STAGES = ("analyze", "flag_issues", "final_decision")
TASKS = ("easy", "medium", "hard")


class InvoiceEnvironment:
    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            seed_value = os.getenv("INVOICE_ENV_SEED")
            seed = int(seed_value) if seed_value is not None else None

        self._step_count = 0
        self._current_task: Optional[TaskRecord] = None
        self._episode_done = False
        self._rng = random.Random(seed)
        self.stage = STAGES[0]
        self.task = TASKS[0]
        self._stage_index = 0
        self._previous_findings: List[str] = []

    def _normalize_task(
        self,
        task: Optional[str] = None,
        difficulty: Optional[Difficulty] = None,
    ) -> Difficulty:
        candidate = task if task is not None else difficulty
        if candidate in TASKS:
            return cast(Difficulty, candidate)
        return cast(Difficulty, self._rng.choice(list(TASKS)))

    def _vendor_name(self, invoice: Dict[str, Any]) -> str:
        category = str(invoice.get("category", "general")).replace("_", " ").strip().title() or "General"
        return f"{category} Vendor"

    def _easy_invoice(self, invoice: Dict[str, Any], decision: str) -> Dict[str, Any]:
        amount = round(float(invoice.get("amount", 0.0)), 2)
        enriched = dict(invoice)
        enriched["vendor_name"] = self._vendor_name(invoice)
        enriched["line_items"] = [
            {
                "description": str(invoice.get("description", "Invoice item")),
                "quantity": 1,
                "unit_price": amount,
                "line_total": amount,
            }
        ]
        enriched["subtotal"] = amount
        enriched["tax_amount"] = 0.0
        enriched["computed_total"] = amount
        enriched["reported_total"] = amount if decision == "approve" else round(amount + 12.5, 2)
        enriched["missing_fields"] = [] if decision == "approve" else ["manager_approval"]
        enriched["anomaly_flags"] = [] if decision == "approve" else [
            "reported total does not match subtotal",
            "manager approval field is missing",
        ]
        return enriched

    def _medium_invoice(self, invoice: Dict[str, Any], decision: str) -> Dict[str, Any]:
        amount = round(float(invoice.get("amount", 0.0)), 2)
        first = round(amount * 0.58, 2)
        second = round(amount - first, 2)
        subtotal = round(first + second, 2)
        tax_rate = 0.07
        correct_tax = round(subtotal * tax_rate, 2)
        reported_tax = correct_tax if decision == "approve" else round(correct_tax + 0.03, 2)
        computed_total = round(subtotal + correct_tax, 2)
        reported_total = computed_total if decision == "approve" else round(subtotal + reported_tax, 2)
        enriched = dict(invoice)
        enriched["vendor_name"] = f"{self._vendor_name(invoice)} Services"
        enriched["line_items"] = [
            {
                "description": "Primary service charge",
                "quantity": 1,
                "unit_price": first,
                "line_total": first,
            },
            {
                "description": "Supporting fee",
                "quantity": 1,
                "unit_price": second,
                "line_total": second,
            },
        ]
        enriched["subtotal"] = subtotal
        enriched["tax_rate"] = tax_rate
        enriched["tax_amount"] = reported_tax
        enriched["expected_tax_amount"] = correct_tax
        enriched["computed_total"] = computed_total
        enriched["reported_total"] = reported_total
        enriched["rounding_adjustment"] = 0.0 if decision == "approve" else round(reported_total - computed_total, 2)
        enriched["anomaly_flags"] = [] if decision == "approve" else [
            "tax amount does not match subtotal",
            "reported total has a subtle rounding mismatch",
        ]
        return enriched

    def _hard_invoice(self, invoice: Dict[str, Any], decision: str) -> Dict[str, Any]:
        amount = round(float(invoice.get("amount", 0.0)), 2)
        first = round(amount * 0.34, 2)
        second = round(amount * 0.33, 2)
        third = round(amount - first - second, 2)
        line_items = [
            {
                "description": "Core service package",
                "quantity": 1,
                "unit_price": first,
                "line_total": first,
            },
            {
                "description": "Implementation support",
                "quantity": 1,
                "unit_price": second,
                "line_total": second,
            },
            {
                "description": "Compliance review",
                "quantity": 1,
                "unit_price": third,
                "line_total": third,
            },
        ]
        subtotal = round(sum(item["line_total"] for item in line_items), 2)
        tax_rate = 0.08
        expected_tax = round(subtotal * tax_rate, 2)
        reported_tax = expected_tax if decision == "approve" else round(expected_tax + 1.11, 2)
        computed_total = round(subtotal + expected_tax, 2)
        reported_total = computed_total if decision == "approve" else round(computed_total + 18.75, 2)
        enriched = dict(invoice)
        enriched["vendor_name"] = "Verified Strategic Partners LLC" if decision == "approve" else "Strategic Partners LLC"
        enriched["vendor_registered_name"] = "Verified Strategic Partners LLC" if decision == "approve" else "Unknown Strategic Partner"
        enriched["vendor_status"] = "verified" if decision == "approve" else "unverified"
        enriched["purchase_order"] = "PO-2048" if decision == "approve" else "PO-2048-DRAFT"
        enriched["department"] = "operations" if decision == "approve" else "operations"
        enriched["approver_department"] = "operations" if decision == "approve" else "finance"
        enriched["line_items"] = line_items
        enriched["subtotal"] = subtotal
        enriched["tax_rate"] = tax_rate
        enriched["tax_amount"] = reported_tax
        enriched["expected_tax_amount"] = expected_tax
        enriched["computed_total"] = computed_total
        enriched["reported_total"] = reported_total
        enriched["service_period"] = {
            "start": str(invoice.get("date", "")),
            "end": str(invoice.get("date", "")),
        }
        enriched["anomaly_flags"] = [] if decision == "approve" else [
            "vendor registration does not match vendor name",
            "approver department conflicts with submitting department",
            "reported total conflicts with computed line-item total",
        ]
        return enriched

    def _task_keywords(self, task: Difficulty, decision: str, invoice: Dict[str, Any]) -> List[str]:
        if task == "easy":
            if decision == "approve":
                return ["clear totals", "single line item", "receipt present", "obvious valid invoice"]
            return ["obvious total mismatch", "missing approval", "clear invalid invoice"]
        if task == "medium":
            if decision == "approve":
                return ["tax calculation matches subtotal", "reported total is consistent", "subtle but valid invoice"]
            return ["tax mismatch", "rounding error", "slight total mismatch"]
        if decision == "approve":
            return ["multiple line items are consistent", "verified vendor", "cross-field checks align", "complex but valid invoice"]
        return ["vendor anomaly", "conflicting departments", "multi-step total conflict", "complex invoice inconsistency"]

    def _enrich_task_record(self, task_record: TaskRecord, task: Difficulty) -> TaskRecord:
        base_invoice = dict(task_record.invoice)
        decision = str(task_record.decision)
        if task == "easy":
            invoice = self._easy_invoice(base_invoice, decision)
        elif task == "medium":
            invoice = self._medium_invoice(base_invoice, decision)
        else:
            invoice = self._hard_invoice(base_invoice, decision)
        keywords = list(task_record.keywords) + self._task_keywords(task, decision, invoice)
        return TaskRecord(invoice=invoice, decision=task_record.decision, keywords=keywords)

    def _build_observation(self) -> Observation:
        invoice = dict(self._current_task.invoice) if self._current_task else {}
        return Observation(
            stage=self.stage,
            invoice=invoice,
            previous_findings=list(self._previous_findings),
        )

    def reset(
        self,
        difficulty: Optional[Difficulty] = None,
        seed: Optional[int] = None,
        task: Optional[str] = None,
    ) -> Observation:
        if seed is not None:
            self._rng = random.Random(seed)

        chosen_task = self._normalize_task(task=task, difficulty=difficulty)
        self.task = chosen_task
        base_task = get_random_task(chosen_task, rng=self._rng)
        self._current_task = self._enrich_task_record(base_task, chosen_task)
        self._step_count = 0
        self._episode_done = False
        self._stage_index = 0
        self.stage = STAGES[self._stage_index]
        self._previous_findings = []
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        if self._current_task is None:
            raise RuntimeError("Environment must be reset before calling step().")
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset().")

        expected_stage = STAGES[self._stage_index]
        reward = grade(action, self._current_task, self.task, expected_stage, self._previous_findings)
        info = build_feedback(
            action,
            self._current_task,
            self.task,
            expected_stage,
            self._previous_findings,
            reward,
        )

        captured_findings = list(info.get("captured_findings") or [])
        for finding in captured_findings:
            finding_text = str(finding).strip()
            if finding_text and finding_text not in self._previous_findings:
                self._previous_findings.append(finding_text)

        self._step_count += 1
        if self._stage_index < len(STAGES) - 1:
            self._stage_index += 1
            self.stage = STAGES[self._stage_index]
            done = False
        else:
            self._episode_done = True
            done = True

        info["task"] = self.task
        info["completed_stage"] = expected_stage
        info["next_stage"] = None if done else self.stage
        info["step_count"] = self._step_count
        info["previous_findings"] = list(self._previous_findings)

        return StepResult(
            observation=self._build_observation(),
            reward=reward,
            done=done,
            info=info,
        )

    def state(self) -> State:
        current_invoice = dict(self._current_task.invoice) if self._current_task else {}
        return State(
            step_count=self._step_count,
            current_invoice=current_invoice,
            stage=self.stage if self._current_task is not None else None,
            previous_findings=list(self._previous_findings),
        )