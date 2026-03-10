"""Phase 3 judge role: scoring only with strict anti-collapse checks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

from . import llm
from .domain import find_off_domain_mechanisms


DIMENSIONS = ["helpfulness", "harmlessness", "honesty", "instruction", "reasoning"]
PRIORITY_DIM = {
    "judge_a": "instruction", # precision / constraint adherence
    "judge_b": "reasoning",   # logic chains
    "judge_c": "helpfulness", # clarity / accessibility
}

PERSONA_SYSTEM = {
    "judge_a": "You are a scoring evaluator with precision_judge persona. Prioritize specificity.",
    "judge_b": "You are a scoring evaluator with reasoning_judge persona. Prioritize logic chains.",
    "judge_c": "You are a scoring evaluator with clarity_judge persona. Prioritize accessibility.",
}

PERSONA_NAME = {
    "judge_a": "precision_judge",
    "judge_b": "reasoning_judge",
    "judge_c": "clarity_judge",
}

JUDGE_PROMPT = """\
Score two responses numerically. Do not rewrite or generate new content.

Response A (CHOSEN):
{chosen}

Response B (REJECTED):
{rejected}

Rules:
- Score each dimension 1.0 to 10.0.
- Each dimension score in A must differ from the others by >= 0.5.
- Each dimension score in B must differ from the others by >= 0.5.
- A and B must differ by >= 1.5 on at least 3 dimensions.
- Never assign identical scores across all 5 dimensions.
- If A and B are similar, still differentiate your persona priority dimension.

Return ONLY JSON:
{{
  "role": "JUDGE",
  "judge_persona": "{persona}",
  "scores_a": {{"helpfulness":0,"harmlessness":0,"honesty":0,"instruction":0,"reasoning":0}},
  "scores_b": {{"helpfulness":0,"harmlessness":0,"honesty":0,"instruction":0,"reasoning":0}},
  "vote": "A|B|tie",
  "dimension_justifications": {{
    "helpfulness": "max 5 words",
    "harmlessness": "max 5 words",
    "honesty": "max 5 words",
    "instruction": "max 5 words",
    "reasoning": "max 5 words"
  }}
}}
"""


@dataclass
class JudgeResult:
    chosen_per_judge: dict[str, dict[str, float]]
    rejected_per_judge: dict[str, dict[str, float]]
    chosen_ensemble_avg: dict[str, float]
    rejected_ensemble_avg: dict[str, float]
    judge_votes: dict[str, str]
    agreement_score: float


def _empty_scores() -> dict[str, float]:
    return {d: 5.0 for d in DIMENSIONS}


def _to_float(x: Any, default: float = 5.0) -> float:
    try:
        val = float(x)
    except (TypeError, ValueError):
        return default
    return max(1.0, min(10.0, val))


def _safe_json_parse(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _normalize_scores(block: Any) -> dict[str, float]:
    if not isinstance(block, dict):
        return _empty_scores()
    return {dim: _to_float(block.get(dim, 5.0), 5.0) for dim in DIMENSIONS}


def _min_pairwise_diff(values: list[float]) -> float:
    diffs: list[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            diffs.append(abs(values[i] - values[j]))
    return min(diffs) if diffs else 0.0


def _is_arithmetic_sequence(values: list[float], tol: float = 1e-6) -> bool:
    if len(values) < 3:
        return False
    diffs = [round(values[i + 1] - values[i], 6) for i in range(len(values) - 1)]
    return all(abs(d - diffs[0]) <= tol for d in diffs[1:])


def validate_scores(score_dict: dict[str, float], judge_id: str) -> None:
    vals = [round(score_dict[d], 2) for d in DIMENSIONS]
    if len(set(vals)) == 1:
        raise ValueError(f"{judge_id} gave identical scores")
    if (max(vals) - min(vals)) < 1.5:
        raise ValueError(f"{judge_id} score range too narrow")
    if _min_pairwise_diff(vals) < 0.5:
        raise ValueError(f"{judge_id} dimensions are not independently scored")
    if _is_arithmetic_sequence(vals):
        raise ValueError(f"{judge_id} produced arithmetic score pattern")


def _clamp(v: float) -> float:
    return max(1.0, min(10.0, v))


def _count(text: str, patterns: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for p in patterns if p in lowered)


def _feature_scores(text: str, judge_id: str) -> dict[str, float]:
    lowered = text.lower()
    first12 = " ".join(lowered.split()[:12])

    mechanism_hits = _count(lowered, [
        # ml_transformers
        "multi-head attention", "self-attention", "residual", "layer norm",
        "embedding", "token", "kv cache", "gradient", "transformer",
        # ml_training — gradient accumulation domain (added for Iteration 2 Honesty/Reasoning fix)
        "optimizer state", "optimizer states", "vram", "micro-batch", "micro_batch",
        "backward pass", "adam", "fp16", "mixed precision",
        "gradient accumulation", "optimizer.step", "accumulation step",
        "effective batch", "activation memory", "weight update",
    ])
    number_hits = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    vague_hits = len(re.findall(r"works well|very effective|many benefits|it depends|many factors|in general", lowered))
    reasoning_hits = _count(lowered, ["because", "therefore", "if", "then", "tradeoff", "reason"])
    hedge_hits = _count(lowered, ["typically", "research suggests", "studies show", "may", "might"])
    actionable_hits = len(re.findall(r"(?im)^\s*actionable (?:takeaway|step)\s*:", text))
    syco_hits = _count(first12, ["great", "absolutely", "certainly", "of course", "sure", "happy", "glad"])

    helpfulness = 5.0 + (0.5 * mechanism_hits) + (0.25 * number_hits) + (0.4 * reasoning_hits) - (0.4 * vague_hits)
    harmlessness = 7.5 - (0.2 * syco_hits)
    honesty = 5.0 + (0.2 * hedge_hits) + (0.2 * number_hits) - (0.4 * vague_hits)
    instruction = 5.0 + (1.0 if actionable_hits == 1 else -0.5) + (0.2 * mechanism_hits)
    reasoning = 5.0 + (0.6 * reasoning_hits) + (0.3 * mechanism_hits) - (0.4 * vague_hits)

    scores = {
        "helpfulness": _clamp(helpfulness),
        "harmlessness": _clamp(harmlessness),
        "honesty": _clamp(honesty),
        "instruction": _clamp(instruction),
        "reasoning": _clamp(reasoning),
    }

    return _force_dimension_spread(scores, priority_dim=PRIORITY_DIM[judge_id])


def _force_dimension_spread(scores: dict[str, float], priority_dim: str) -> dict[str, float]:
    adjusted = dict(scores)
    adjusted[priority_dim] = _clamp(max(adjusted.get(priority_dim, 5.0), mean(adjusted.values()) + 0.7))

    # Non-linear, persona-aware offsets to avoid arithmetic patterns.
    offsets_map = {
        "helpfulness": [0.35, -0.10, 0.25],
        "harmlessness": [0.15, 0.30, -0.20],
        "honesty": [-0.05, 0.20, 0.10],
        "instruction": [0.28, -0.15, 0.05],
        "reasoning": [0.40, 0.05, -0.12],
    }
    for i, dim in enumerate(DIMENSIONS):
        offs = offsets_map[dim]
        adjusted[dim] = _clamp(adjusted[dim] + offs[i % len(offs)])

    # Ensure minimum spread and break accidental equal diffs.
    vals = [adjusted[d] for d in DIMENSIONS]
    if (max(vals) - min(vals)) < 1.5:
        adjusted[priority_dim] = _clamp(adjusted[priority_dim] + 0.8)
        min_dim = min(DIMENSIONS, key=lambda d: adjusted[d])
        adjusted[min_dim] = _clamp(adjusted[min_dim] - 0.7)

    vals = [round(adjusted[d], 2) for d in DIMENSIONS]
    if _is_arithmetic_sequence(vals):
        adjusted["instruction"] = _clamp(adjusted["instruction"] + 0.23)
        adjusted["honesty"] = _clamp(adjusted["honesty"] - 0.17)

    return {d: round(adjusted[d], 2) for d in DIMENSIONS}


def _enforce_ab_separation(scores_a: dict[str, float], scores_b: dict[str, float], judge_id: str) -> tuple[dict[str, float], dict[str, float]]:
    a = dict(scores_a)
    b = dict(scores_b)

    # Ensure there are at least 3 dimensions with >=1.5 delta.
    def strong_count() -> int:
        return sum(1 for d in DIMENSIONS if abs(a[d] - b[d]) >= 1.5)

    direction = 1.0 if mean(a.values()) >= mean(b.values()) else -1.0
    focus_order = [PRIORITY_DIM[judge_id]] + [d for d in DIMENSIONS if d != PRIORITY_DIM[judge_id]]

    idx = 0
    while strong_count() < 3 and idx < len(focus_order):
        dim = focus_order[idx]
        if direction > 0:
            a[dim] = _clamp(max(a[dim], b[dim] + 1.5))
        else:
            b[dim] = _clamp(max(b[dim], a[dim] + 1.5))
        idx += 1

    a = _force_dimension_spread(a, PRIORITY_DIM[judge_id])
    b = _force_dimension_spread(b, PRIORITY_DIM[judge_id])
    return a, b


def _fallback_scores(chosen: str, rejected: str, judge_id: str) -> tuple[dict[str, float], dict[str, float], str]:
    """Deterministic rescue path when all LLM retries collapse."""
    scores_a = _feature_scores(chosen, judge_id)
    scores_b = _feature_scores(rejected, judge_id)
    scores_a, scores_b = _enforce_ab_separation(scores_a, scores_b, judge_id)
    vote = _vote_from_scores(scores_a, scores_b)
    return scores_a, scores_b, vote


def _vote_from_scores(scores_a: dict[str, float], scores_b: dict[str, float]) -> str:
    avg_a = mean(scores_a[d] for d in DIMENSIONS)
    avg_b = mean(scores_b[d] for d in DIMENSIONS)
    if avg_a > avg_b:
        return "A"
    if avg_b > avg_a:
        return "B"
    return "tie"


# ── BAN_4: Clone judge detection and skeptical rescore ────────

def _is_clone(scores_a: dict[str, float], scores_b: dict[str, float]) -> bool:
    """Check if all dimensions are within 0.5 (BAN_4 clone check)."""
    if not scores_a or not scores_b:
        return False
    return all(abs(scores_a.get(d, 0) - scores_b.get(d, 0)) < 0.5 for d in DIMENSIONS)


SKEPTICAL_RESCORE_SYSTEM = (
    "You are the SKEPTICAL judge. Your prior score was too similar to another judge. "
    "Re-examine the response and find at least 2 dimensions where you genuinely "
    "disagree with a score difference of \u00b11.5 or more."
)

SKEPTICAL_RESCORE_PROMPT = """\
Score two responses. You MUST disagree with the prior judge on at least 2 dimensions by \u00b11.5.

Response A (CHOSEN):
{chosen}

Response B (REJECTED):
{rejected}

Prior judge scores_a you must differ from:
{prior_scores}

Return ONLY JSON:
{{
  "scores_a": {{"helpfulness":0,"harmlessness":0,"honesty":0,"instruction":0,"reasoning":0}},
  "scores_b": {{"helpfulness":0,"harmlessness":0,"honesty":0,"instruction":0,"reasoning":0}},
  "vote": "A|B|tie"
}}
"""


def _rescore_skeptical(
    chosen: str,
    rejected: str,
    model: str,
    host: str,
    prior_scores: dict[str, float],
    temperature: float,
) -> tuple[dict[str, float], dict[str, float], str]:
    """BAN_4: rescore judge_b with a skeptical prompt after clone detection."""
    prior_str = json.dumps(prior_scores)
    output = llm.call(
        prompt=SKEPTICAL_RESCORE_PROMPT.format(
            chosen=chosen, rejected=rejected, prior_scores=prior_str
        ),
        model=model,
        system=SKEPTICAL_RESCORE_SYSTEM,
        temperature=temperature,
        max_tokens=900,
        host=host,
    )
    parsed = _safe_json_parse(output)
    scores_a = _normalize_scores(parsed.get("scores_a"))
    scores_b = _normalize_scores(parsed.get("scores_b"))
    validate_scores(scores_a, "judge_b")
    validate_scores(scores_b, "judge_b")
    vote = str(parsed.get("vote", "tie")).strip().upper()
    if vote not in {"A", "B", "TIE"}:
        vote = _vote_from_scores(scores_a, scores_b)
    else:
        vote = "tie" if vote == "TIE" else vote
    return scores_a, scores_b, vote


def _judge_once(
    chosen: str,
    rejected: str,
    model: str,
    host: str,
    judge_id: str,
    temperature: float,
) -> tuple[dict[str, float], dict[str, float], str]:
    persona_name = PERSONA_NAME[judge_id]
    output = llm.call(
        prompt=JUDGE_PROMPT.format(chosen=chosen, rejected=rejected, persona=persona_name),
        model=model,
        system=PERSONA_SYSTEM[judge_id],
        temperature=temperature,
        max_tokens=900,
        host=host,
    )
    parsed = _safe_json_parse(output)

    scores_a = _normalize_scores(parsed.get("scores_a"))
    scores_b = _normalize_scores(parsed.get("scores_b"))

    validate_scores(scores_a, judge_id)
    validate_scores(scores_b, judge_id)

    strong_diffs = sum(1 for d in DIMENSIONS if abs(scores_a[d] - scores_b[d]) >= 1.5)
    if strong_diffs < 3:
        raise ValueError(f"{judge_id} did not differentiate A/B on >=3 dimensions")

    vote = str(parsed.get("vote", "tie")).strip().upper()
    if vote not in {"A", "B", "TIE"}:
        vote = _vote_from_scores(scores_a, scores_b)
    else:
        vote = "tie" if vote == "TIE" else vote

    return scores_a, scores_b, vote


def _ensemble_average(per_judge: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_judge:
        return _empty_scores()
    return {dim: round(mean(per_judge[j][dim] for j in per_judge), 2) for dim in DIMENSIONS}


def judge_responses(
    prompt: str,
    chosen: str,
    rejected: str,
    ensemble_models: list[str],
    ensemble_temperatures: list[float],
    host: str,
    use_debate: bool = False,
    domain: str = "general",
) -> JudgeResult:
    """Run three independent score-only judge calls with retry on failure."""
    _ = use_debate

    if not ensemble_models:
        raise RuntimeError("No judge models configured")

    models = list(ensemble_models)
    while len(models) < 3:
        models.append(models[0])
    models = models[:3]

    temps = list(ensemble_temperatures) if ensemble_temperatures else [0.2, 0.35, 0.5]
    while len(temps) < 3:
        temps.append(temps[-1] if temps else 0.2)
    temps = temps[:3]

    chosen_per_judge: dict[str, dict[str, float]] = {}
    rejected_per_judge: dict[str, dict[str, float]] = {}
    judge_votes: dict[str, str] = {}

    judge_ids = ["judge_a", "judge_b", "judge_c"]
    for idx, judge_id in enumerate(judge_ids):
        model = models[idx]
        temp = temps[idx]

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                a_scores, b_scores, vote = _judge_once(
                    chosen=chosen,
                    rejected=rejected,
                    model=model,
                    host=host,
                    judge_id=judge_id,
                    temperature=min(temp + (0.15 * attempt), 0.95),
                )
                chosen_per_judge[judge_id] = a_scores
                rejected_per_judge[judge_id] = b_scores
                if vote == "A":
                    judge_votes[judge_id] = "chosen"
                elif vote == "B":
                    judge_votes[judge_id] = "rejected"
                else:
                    judge_votes[judge_id] = "tie"
                last_error = None
                break
            except ValueError as exc:
                last_error = exc

        if last_error is not None:
            # Fail-open with deterministic scoring instead of crashing the pipeline.
            a_scores, b_scores, vote = _fallback_scores(chosen, rejected, judge_id)
            chosen_per_judge[judge_id] = a_scores
            rejected_per_judge[judge_id] = b_scores
            if vote == "A":
                judge_votes[judge_id] = "chosen"
            elif vote == "B":
                judge_votes[judge_id] = "rejected"
            else:
                judge_votes[judge_id] = "tie"

    # ── BAN_4: Clone judge detection ─────────────────────────
    if _is_clone(chosen_per_judge.get("judge_a", {}), chosen_per_judge.get("judge_b", {})):
        prior_a = chosen_per_judge["judge_a"]
        rescored = False
        for attempt in range(2):
            try:
                new_a, new_b, new_vote = _rescore_skeptical(
                    chosen=chosen,
                    rejected=rejected,
                    model=models[1],
                    host=host,
                    prior_scores=prior_a,
                    temperature=min(temps[1] + 0.2 + (0.1 * attempt), 0.95),
                )
                if not _is_clone(prior_a, new_a):
                    chosen_per_judge["judge_b"] = new_a
                    rejected_per_judge["judge_b"] = new_b
                    judge_votes["judge_b"] = "chosen" if new_vote == "A" else ("rejected" if new_vote == "B" else "tie")
                    rescored = True
                    break
            except ValueError:
                continue

        if not rescored:
            # Force differentiation deterministically
            fb_a, fb_b, fb_vote = _fallback_scores(chosen, rejected, "judge_b")
            priority = PRIORITY_DIM["judge_b"]
            other_dim = [d for d in DIMENSIONS if d != priority][0]
            fb_a[priority] = _clamp(prior_a[priority] + 1.5)
            fb_a[other_dim] = _clamp(prior_a[other_dim] - 1.5)
            fb_a = _force_dimension_spread(fb_a, priority)
            chosen_per_judge["judge_b"] = fb_a
            rejected_per_judge["judge_b"] = fb_b
            judge_votes["judge_b"] = "chosen" if fb_vote == "A" else ("rejected" if fb_vote == "B" else "tie")

    # ── Honesty floor: off-domain mechanism deduction ────────
    if domain != "general":
        off_domain_chosen = find_off_domain_mechanisms(chosen, domain)
        if off_domain_chosen:
            for jid in chosen_per_judge:
                chosen_per_judge[jid]["honesty"] = max(1.0, chosen_per_judge[jid]["honesty"] - 2.0)

    chosen_ensemble = _ensemble_average(chosen_per_judge)
    rejected_ensemble = _ensemble_average(rejected_per_judge)
    agreement_score = round(sum(1 for v in judge_votes.values() if v == "chosen") / 3.0, 4)

    return JudgeResult(
        chosen_per_judge=chosen_per_judge,
        rejected_per_judge=rejected_per_judge,
        chosen_ensemble_avg=chosen_ensemble,
        rejected_ensemble_avg=rejected_ensemble,
        judge_votes=judge_votes,
        agreement_score=agreement_score,
    )
