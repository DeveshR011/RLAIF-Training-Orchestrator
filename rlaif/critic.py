"""Phase 2 critic role: violations + fixes + deterministic recheck."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from . import llm
from .domain import find_off_domain_mechanisms


BANNED_WORDS = [
    "happy", "glad", "great", "certainly", "absolutely", "of course", "sure",
    "delighted", "wonderful", "fantastic", "excellent", "indeed", "definitely",
]

NAMED_TOKENS = [
    # ml_transformers
    "multi-head attention", "self-attention", "residual connections", "layer normalization",
    "gradient checkpointing", "tokenization", "embeddings", "rnn", "cnn", "transformer",
    "kv cache", "beam search", "adam", "dropout",
    # ml_training — gradient accumulation domain (added for Iteration 2 C3 fix)
    "optimizer state", "optimizer states", "vram", "micro-batch", "micro_batch",
    "backward pass", "fp16", "mixed precision", "zero", "gradient accumulation",
    "optimizer.step", "accumulation step", "accumulation steps",
    "effective batch", "activation memory", "weight update",
]


@dataclass
class CritiqueResult:
    violations_found: list[str]
    unresolved_violations: list[str] | None = None
    revised_response: str = ""
    violations_fixed: list[str] | None = None
    violations_unresolvable: list[str] | None = None
    recheck_passed: bool = False
    sycophancy_score: float = 0.0
    sycophancy_detected: bool = False
    pattern_matched: str | None = None
    constitution_version: str = "2.0"


CRITIC_SYSTEM = """\
You are a strict constitutional critic.
Find violations and fix them. Return JSON only.
"""

CRITIC_PROMPT = """\
Audit and repair this response for C1-C6:

C1 ANTI-SYCOPHANCY: banned words in first 12 words.
C2 ANTI-HALLUCINATION: unattributed factual claims.
C3 ANTI-VAGUENESS: each paragraph needs named concept/number/comparison/technique.
C4 CALIBRATED UNCERTAINTY: estimates should use hedges.
C5 USER AUTONOMY: avoid "you must" directives without reasoning.
C6 ANTI-DUPLICATE-TAKEAWAY: exactly one actionable step at the end.

Response:
{response}

Return ONLY JSON:
{{
  "violations": ["C1","C2","C3","C4","C5","C6"],
  "clean_version": "..."
}}
"""


def _safe_json_parse(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def _first_n_words(text: str, n: int) -> str:
    return " ".join(text.strip().lower().split()[:n])


def _contains_banned(head_or_text: str) -> str | None:
    lowered = head_or_text.lower()
    for phrase in BANNED_WORDS:
        # Match exact words/phrases to avoid substring false positives (e.g., "architecture" -> "sure").
        pattern = r"\b" + r"\s+".join(re.escape(tok) for tok in phrase.split()) + r"\b"
        if re.search(pattern, lowered):
            return phrase
    return None


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _contains_named_requirement(paragraph: str) -> bool:
    lowered = paragraph.lower()
    has_named = any(tok in lowered for tok in NAMED_TOKENS)
    has_number = bool(re.search(r"\b\d+(?:\.\d+)?\b", paragraph))
    has_comparison = " vs " in lowered or "compared to" in lowered
    has_technique = any(tok in lowered for tok in ["technique", "method", "algorithm", "mechanism"])
    return has_named or has_number or has_comparison or has_technique


def _detect_violations(text: str, domain: str = "general") -> list[str]:
    violations: list[str] = []
    lowered = text.lower()

    # C1
    first12 = _first_n_words(text, 12)
    if _contains_banned(first12) is not None:
        violations.append("C1")

    # C2
    has_unattributed_numeric = bool(re.search(r"\b\d+(?:\.\d+)?%?\b", text)) and not any(
        s in lowered for s in ["according to", "research suggests", "studies show", "typically", "source", "paper", "study"]
    )
    named_study_without_context = bool(re.search(r"\b(study|paper|report)\b", lowered)) and not any(
        s in lowered for s in ["according to", "published", "in", "journal", "conference"]
    )
    if has_unattributed_numeric or named_study_without_context:
        violations.append("C2")

    # C2_domain: off-domain mechanism contamination (BAN_1 at violation level)
    if domain != "general":
        off_domain = find_off_domain_mechanisms(text, domain)
        if off_domain:
            violations.append("C2_domain")

    # C3
    paragraphs = _split_paragraphs(text)
    content_paragraphs = [p for p in paragraphs if not re.match(r"(?im)^\s*actionable (?:takeaway|step)\s*:", p.strip())]
    if any(not _contains_named_requirement(p) for p in content_paragraphs):
        violations.append("C3")

    # C4
    estimate_like = any(k in lowered for k in ["about", "around", "roughly", "estimate", "approximately"])
    has_hedge = any(k in lowered for k in ["typically", "research suggests", "may", "might", "often"])
    if estimate_like and not has_hedge:
        violations.append("C4")

    # C5
    if "you must" in lowered and "because" not in lowered:
        violations.append("C5")

    # C6
    takeaway_lines = re.findall(r"(?im)^\s*actionable (?:takeaway|step)\s*:", text)
    if len(takeaway_lines) != 1:
        violations.append("C6")

    return list(dict.fromkeys(violations))


def _ensure_single_takeaway(text: str) -> str:
    lines = text.strip().splitlines()
    kept: list[str] = []
    takeaway_value = None
    for ln in lines:
        if re.match(r"(?im)^\s*actionable (?:takeaway|step)\s*:", ln):
            if takeaway_value is None:
                takeaway_value = ln.split(":", 1)[1].strip() if ":" in ln else "Execute one concrete next step."
            continue
        kept.append(ln)
    if takeaway_value is None:
        takeaway_value = "Execute one concrete next step."
    rebuilt = "\n".join(kept).strip()
    return (rebuilt + "\n\nActionable step: " + takeaway_value).strip()


def _sycophancy_probe(text: str) -> tuple[bool, float, str | None]:
    lowered = text.lower()
    first12 = _first_n_words(text, 12)

    hit_first = _contains_banned(first12)
    if hit_first:
        return True, 1.0, hit_first

    hit_any = _contains_banned(lowered)
    if hit_any:
        return True, 0.5, hit_any

    return False, 0.0, None


def critique_response(
    prompt: str,
    response: str,
    constitution: dict[str, Any],
    model: str,
    host: str,
    probe: str | None = None,
    apply_probe: bool = True,
    domain: str = "general",
) -> CritiqueResult:
    """Audit C1-C6, rewrite, and recheck deterministically."""
    _ = (prompt, constitution, probe, apply_probe)

    violations_found = _detect_violations(response, domain=domain)

    output = llm.call(
        prompt=CRITIC_PROMPT.format(response=response),
        model=model,
        system=CRITIC_SYSTEM,
        temperature=0.1,
        max_tokens=1000,
        host=host,
    )
    parsed = _safe_json_parse(output)

    # Never trust model-reported violation labels for bookkeeping.
    # Use deterministic detector to avoid phantom metadata.

    clean_version = str(parsed.get("clean_version", "")).strip() if isinstance(parsed, dict) else ""
    if not clean_version:
        clean_version = response.strip()

    clean_version = _ensure_single_takeaway(clean_version)

    # Re-check gate
    unresolved = _detect_violations(clean_version, domain=domain)
    fixed = [v for v in violations_found if v not in unresolved]
    unresolvable = [v for v in violations_found if v in unresolved]

    # Consistency gate: fixed must be strict subset of found.
    fixed = [v for v in fixed if v in violations_found]
    if not violations_found:
        fixed = []
        unresolvable = []

    recheck_passed = len(unresolved) == 0

    syco_detected, syco_score, pattern = _sycophancy_probe(clean_version)

    # Mandatory consistency
    if syco_score >= 0.5:
        syco_detected = True
    else:
        syco_detected = False

    return CritiqueResult(
        violations_found=violations_found,
        unresolved_violations=unresolved,
        revised_response=clean_version,
        violations_fixed=fixed,
        violations_unresolvable=unresolvable,
        recheck_passed=recheck_passed,
        sycophancy_score=syco_score,
        sycophancy_detected=syco_detected,
        pattern_matched=pattern,
        constitution_version="2.0",
    )


def verify_rejected_has_flaws(
    rejected: str,
    constitution: dict[str, Any],
    model: str,
    host: str,
) -> list[str]:
    """Verify rejected contains sycophantic opener and vague non-answer."""
    _ = (constitution, model, host)
    detected: list[str] = []

    first12 = _first_n_words(rejected, 12)
    if _contains_banned(first12) is not None:
        detected.append("sycophantic_opener")

    lowered = rejected.lower()
    vague_hits = len(re.findall(r"works well|very effective|many benefits|in general|it depends|many factors", lowered))
    has_specific = bool(re.search(r"\b\d+(?:\.\d+)?\b", rejected)) or any(tok in lowered for tok in NAMED_TOKENS)
    if vague_hits >= 2 and not has_specific:
        detected.append("vague_non_answer")

    return detected
