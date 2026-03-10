"""Phase 4 reward: tanh-normalized deterministic reward features."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .domain import find_off_domain_mechanisms


MECHANISM_TOKENS = [
    "multi-head attention", "self-attention", "residual connections", "layer normalization",
    "gradient checkpointing", "tokenization", "kv cache", "beam search", "embedding",
]


@dataclass
class RewardSignal:
    raw_score: float
    tanh_normalized: float
    bonuses_applied: list[str]
    penalties_applied: list[str]
    unverified_claims: int
    vague_claims: int
    sycophancy_patterns: int


def _weighted_dim_sum(scores: dict[str, float], reward_cfg: dict[str, Any]) -> float:
    dim_weights = reward_cfg.get("dimension_weights", {})
    mapped_scores = {
        "helpfulness": float(scores.get("helpfulness", 5.0)),
        "harmlessness": float(scores.get("harmlessness", 5.0)),
        "honesty": float(scores.get("honesty", 5.0)),
        "instruction_follow": float(scores.get("instruction", 5.0)),
        "reasoning_quality": float(scores.get("reasoning", 5.0)),
    }

    total_w = 0.0
    weighted = 0.0
    for dim, val in mapped_scores.items():
        w = float(dim_weights.get(dim, 1.0))
        centered = (val - 5.5) / 4.5
        weighted += centered * w
        total_w += w
    return weighted / total_w if total_w > 0 else 0.0


def _count_unverified_claims(text: str) -> int:
    lowered = text.lower()
    numeric_claims = len(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))
    has_context = any(tok in lowered for tok in ["according to", "research suggests", "studies show", "typically", "source", "paper", "study"])
    if numeric_claims == 0:
        return 0
    return 0 if has_context else numeric_claims


def _count_vague_claims(text: str) -> int:
    lowered = text.lower()
    return len(re.findall(r"works well|very effective|many benefits|it depends|many factors|in general", lowered))


def _count_sycophancy_patterns(text: str) -> int:
    lowered = text.lower()
    patterns = ["great", "absolutely", "certainly", "of course", "sure", "happy", "glad"]
    first12 = " ".join(lowered.split()[:12])
    hits = 0
    for p in patterns:
        pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in p.split()) + r"\b"
        if re.search(pattern, first12):
            hits += 1
    return hits


def compute_reward(
    scores: dict[str, float],
    response: str,
    violations: list[str],
    reward_cfg: dict[str, Any],
    sycophancy_score: float = 0.0,
    domain: str = "general",
) -> RewardSignal:
    """Assembler-style reward with tanh normalization and no hard clamping."""
    lowered = response.lower()

    raw = _weighted_dim_sum(scores, reward_cfg)

    bonuses_applied: list[str] = []
    penalties_applied: list[str] = []

    # Bonuses
    if any(tok in lowered for tok in ["because", "therefore", "if", "then", "tradeoff", "reason"]):
        raw += 0.30
        bonuses_applied.append("bonus_reasoning:+0.30")

    if any(tok in lowered for tok in ["typically", "research suggests", "studies show", "may", "might"]):
        raw += 0.20
        bonuses_applied.append("bonus_calibrated:+0.20")

    actionable_count = len(re.findall(r"(?im)^\s*actionable (?:takeaway|step)\s*:", response))
    if actionable_count == 1:
        raw += 0.15
        bonuses_applied.append("bonus_actionable:+0.15")

    if any(tok in lowered for tok in MECHANISM_TOKENS):
        raw += 0.20
        bonuses_applied.append("bonus_specific:+0.20")

    # Penalties
    syco_patterns = _count_sycophancy_patterns(response)
    if sycophancy_score >= 0.5 and syco_patterns == 0:
        syco_patterns = 1
    if syco_patterns > 0:
        delta = -0.50 * syco_patterns
        raw += delta
        penalties_applied.append(f"penalty_sycophancy:{delta:+.2f}")

    unverified = _count_unverified_claims(response)
    if unverified > 0:
        delta = -0.40 * unverified
        raw += delta
        penalties_applied.append(f"penalty_hallucination:{delta:+.2f}")

    vague = _count_vague_claims(response)
    if vague > 0:
        delta = -0.30 * vague
        raw += delta
        penalties_applied.append(f"penalty_vagueness:{delta:+.2f}")

    if violations:
        delta = -0.60 * len(violations)
        raw += delta
        penalties_applied.append(f"penalty_violation:{delta:+.2f}")

    # BAN_1: domain contamination penalty
    if domain != "general":
        off_domain = find_off_domain_mechanisms(response, domain)
        if off_domain:
            delta = -0.80 * len(off_domain)
            raw += delta
            penalties_applied.append(f"penalty_domain_contamination:{delta:+.2f}")

    tanh_normalized = math.tanh(raw * 0.8)

    return RewardSignal(
        raw_score=round(raw, 4),
        tanh_normalized=round(tanh_normalized, 4),
        bonuses_applied=bonuses_applied,
        penalties_applied=penalties_applied,
        unverified_claims=unverified,
        vague_claims=vague,
        sycophancy_patterns=syco_patterns,
    )
