"""Deterministic validation gate for triplets before dataset persistence."""
from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

from .domain import find_off_domain_mechanisms


# BAN_2: annotation label patterns that should never appear in rejected text
_ANNOTATION_PATTERNS = [
    r"(?im)sycophantic[_\s]*opener\s*:",
    r"(?im)vague[_\s]*non[_\s]*answer\s*:",
    r"(?im)\[flaw\]\s*:?",
    r"(?im)flaw\s*\d*\s*:",
    r"(?im)\[rejected\]\s*:?",
]


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_get(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _assemble_confidence(votes: dict[str, str], score_gap: float) -> tuple[str, float]:
    votes_chosen = sum(1 for v in votes.values() if str(v).lower() == "chosen")
    votes_rejected = sum(1 for v in votes.values() if str(v).lower() == "rejected")

    if votes_chosen == 3:
        preferred = "chosen"
        confidence = 0.85 + (score_gap * 0.03)
    elif votes_chosen == 2:
        preferred = "chosen"
        confidence = 0.60 + (score_gap * 0.05)
    elif votes_rejected == 3:
        preferred = "rejected"
        confidence = 0.85
    elif votes_rejected == 2:
        preferred = "rejected"
        confidence = 0.60
    else:
        preferred = "tie"
        confidence = 0.35

    confidence = min(max(confidence, 0.0), 0.99)
    return preferred, round(confidence, 4)


def _is_arithmetic(values: list[float], tol: float = 1e-6) -> bool:
    if len(values) < 3:
        return False
    diffs = [round(values[i + 1] - values[i], 6) for i in range(len(values) - 1)]
    return all(abs(d - diffs[0]) <= tol for d in diffs[1:])


def _has_annotation_labels(text: str) -> bool:
    """BAN_2: check if rejected text contains self-annotating labels."""
    for pat in _ANNOTATION_PATTERNS:
        if re.search(pat, text):
            return True
    return False


def validate_triplet(
    triplet: dict[str, Any],
    *,
    iteration: int,
    prior_chosen: str | None,
    ensemble_models: list[str],
) -> ValidationResult:
    """Validate logical consistency for a generated triplet."""
    errors: list[str] = []
    warnings: list[str] = []

    chosen = _safe_get(triplet, ["chosen"], {})
    rejected = _safe_get(triplet, ["rejected"], {})
    judge = _safe_get(triplet, ["judge"], {})
    flywheel = _safe_get(triplet, ["flywheel"], {})
    meta = _safe_get(triplet, ["meta"], {})

    expected_diversity = len(set(ensemble_models)) >= 2
    reported_diversity = bool(meta.get("ensemble_diversity", False))
    if reported_diversity != expected_diversity:
        errors.append("meta.ensemble_diversity does not match configured ensemble model diversity")

    detected_flaws = rejected.get("detected_flaws", [])
    if sorted(detected_flaws) != sorted(["sycophantic_opener", "vague_non_answer"]):
        errors.append("rejected.detected_flaws must contain exactly sycophantic_opener and vague_non_answer")

    # BAN_2: rejected must not contain annotation labels
    rejected_text = str(rejected.get("response", ""))
    if _has_annotation_labels(rejected_text):
        errors.append("rejected response contains self-annotating labels (BAN_2)")

    found = chosen.get("constitution_violations", [])
    fixed = chosen.get("violations_fixed", [])
    unresolved = chosen.get("violations_unresolvable", [])

    if len(fixed) > len(found):
        errors.append("violations_fixed cannot be longer than constitution_violations")
    if any(v not in found for v in fixed):
        errors.append("violations_fixed must be a subset of constitution_violations")

    if flywheel.get("next_iteration_seed") != chosen.get("response"):
        errors.append("flywheel.next_iteration_seed must exactly equal chosen.response")

    seed_used = bool(chosen.get("seed_used", False))
    proof = chosen.get("seed_inheritance_proof")
    if iteration > 1 and prior_chosen and not seed_used:
        errors.append("iteration > 1 with provided prior_chosen requires seed_used=true")
    if seed_used and not proof:
        errors.append("seed_used=true requires seed_inheritance_proof")

    votes = judge.get("judge_votes", {})
    if not isinstance(votes, dict):
        errors.append("judge.judge_votes must be an object")
        votes = {}

    allowed = {"chosen", "rejected", "tie"}
    for key in ["judge_a", "judge_b", "judge_c"]:
        val = str(votes.get(key, "tie")).lower()
        if val not in allowed:
            errors.append(f"judge.judge_votes.{key} must be chosen|rejected|tie")

    chosen_votes = sum(1 for v in votes.values() if str(v).lower() == "chosen")
    expected_agreement = round(chosen_votes / 3.0, 4)
    reported_agreement = round(_to_float(judge.get("agreement_score"), 0.0), 4)
    if abs(expected_agreement - reported_agreement) > 0.001:
        errors.append("judge.agreement_score must equal judges_preferring_chosen / 3")

    chosen_avg_block = _safe_get(triplet, ["chosen", "scores", "ensemble_avg"], {})
    rejected_avg_block = _safe_get(triplet, ["rejected", "scores", "ensemble_avg"], {})
    dims = ["helpfulness", "harmlessness", "honesty", "instruction", "reasoning"]
    avg_a = mean(_to_float(chosen_avg_block.get(d), 5.0) for d in dims)
    avg_b = mean(_to_float(rejected_avg_block.get(d), 5.0) for d in dims)

    reported_gap = round(_to_float(judge.get("score_gap"), 0.0), 4)
    expected_gap = round(avg_a - avg_b, 4)
    if abs(reported_gap - expected_gap) > 0.01:
        errors.append("judge.score_gap must equal chosen_avg - rejected_avg")

    expected_pref, expected_conf = _assemble_confidence(votes, reported_gap)
    preferred = str(judge.get("preferred", "tie")).lower()
    reported_conf = round(_to_float(judge.get("confidence"), 0.0), 4)

    if preferred != expected_pref:
        errors.append("judge.preferred is inconsistent with deterministic assembly logic")
    if abs(reported_conf - expected_conf) > 0.001:
        errors.append("judge.confidence is inconsistent with deterministic assembly logic")

    # Anti-pattern judge checks
    judge_vectors: dict[str, tuple[float, float, float, float, float]] = {}
    for judge_id in ["judge_a", "judge_b", "judge_c"]:
        block = _safe_get(triplet, ["chosen", "scores", judge_id], {})
        vals = [_to_float(block.get(d), 5.0) for d in dims]
        judge_vectors[judge_id] = tuple(round(v, 2) for v in vals)
        if _is_arithmetic(vals):
            errors.append(f"{judge_id} produced arithmetic score sequence")
        if (max(vals) - min(vals)) < 1.5:
            errors.append(f"{judge_id} score range < 1.5")
    if len(set(judge_vectors.values())) < 3:
        errors.append("judge outputs are not independent; at least two judges have identical score vectors")

    # BAN_4: clone check (all dims within 0.5)
    judge_a_block = _safe_get(triplet, ["chosen", "scores", "judge_a"], {})
    judge_b_block = _safe_get(triplet, ["chosen", "scores", "judge_b"], {})
    if judge_a_block and judge_b_block:
        all_close = all(
            abs(_to_float(judge_a_block.get(d), 0) - _to_float(judge_b_block.get(d), 0)) < 0.5
            for d in dims
        )
        if all_close:
            errors.append("judge_a and judge_b are clones (all dims within 0.5, BAN_4)")

    flagged = bool(judge.get("flag_for_human_review", False))
    if preferred == "tie" and not flagged:
        errors.append("judge.flag_for_human_review must be true when preferred is tie")
    if reported_conf < 0.75 and not flagged:
        errors.append("judge.flag_for_human_review must be true when confidence < 0.75")

    c_tanh = _to_float(_safe_get(triplet, ["chosen", "reward_breakdown", "tanh_normalized"], 0.0), 0.0)
    r_tanh = _to_float(_safe_get(triplet, ["rejected", "reward_breakdown", "tanh_normalized"], 0.0), 0.0)
    if abs(c_tanh) >= 1.0 or abs(r_tanh) >= 1.0:
        errors.append("reward tanh_normalized must be strictly within (-1, 1)")

    syco = _safe_get(triplet, ["chosen", "sycophancy_probe"], {})
    syco_detected = bool(syco.get("detected", False))
    syco_score = _to_float(syco.get("score", 0.0), 0.0)
    if syco_score >= 0.5 and not syco_detected:
        errors.append("sycophancy_probe.detected must be true when score >= 0.5")
    if syco_score < 0.5 and syco_detected:
        errors.append("sycophancy_probe.detected must be false when score < 0.5")

    if bool(flywheel.get("stop_condition_met", False)):
        errors.append(f"stop condition triggered: {flywheel.get('stop_reason')}")

    # Domain contamination check (BAN_1)
    domain_clean = bool(meta.get("domain_clean", True))
    detected_domain = str(meta.get("detected_domain", "general"))
    off_domain_mechanisms = chosen.get("off_domain_mechanisms", [])
    if off_domain_mechanisms:
        errors.append(f"off-domain mechanisms found in chosen response: {off_domain_mechanisms}")
    if "C2_domain" in found:
        errors.append("C2_domain violation detected: domain contamination in chosen response")

    # GATE_5 verification: score_gap >= 2.0 (BAN_3)
    if reported_gap < 2.0:
        warnings.append(f"GATE_5: score_gap ({reported_gap:.4f}) < 2.0")

    # DPO readiness check (BAN_3 — comprehensive)
    dpo_ready = bool(meta.get("ready_for_dpo_training", False))
    expected_dpo = (
        avg_a >= 7.0
        and reported_gap >= 2.0
        and preferred == "chosen"
        and reported_conf >= 0.75
        and c_tanh > 0.0
        and len(unresolved) == 0
        and syco_score == 0.0
        and domain_clean
    )
    if dpo_ready != expected_dpo:
        errors.append("meta.ready_for_dpo_training is inconsistent with deterministic DPO gate")

    # Gate results consistency
    gates = _safe_get(triplet, ["meta", "gate_results"], None)
    if isinstance(gates, dict):
        all_passed = bool(gates.get("all_passed", False))
        gate_keys = [k for k in gates.keys() if k.startswith("GATE_")]
        computed_all_passed = all(str(gates[k]).lower() == "pass" for k in gate_keys)
        if all_passed != computed_all_passed:
            errors.append("meta.gate_results.all_passed is inconsistent with gate statuses")
        if dpo_ready and not all_passed:
            errors.append("ready_for_dpo_training cannot be true when gate_results.all_passed is false")
        # BAN_3 enforcement: dpo_ready must equal all_passed
        if all_passed != dpo_ready:
            errors.append("BAN_3 violation: all_passed and dpo_ready must always be identical")

    if prior_chosen is None and iteration > 1:
        warnings.append("iteration > 1 without explicit prior_chosen; auto-seed resolution should supply prior")

    return ValidationResult(valid=(len(errors) == 0), errors=errors, warnings=warnings)
