"""Phase 1 generator role with Python-enforced seed injection."""
from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass
import re
from typing import Optional

from . import llm
from .domain import detect_domain, find_off_domain_mechanisms, DOMAIN_FALLBACK_MECHANISM


# ──────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────

@dataclass
class GeneratedPair:
    prompt: str
    chosen: str
    rejected: str
    seed_used: bool = False
    weakest_paragraph_identified: str | None = None
    generation_notes: str = ""
    seed_inheritance_proof: dict | None = None
    iteration: int = 1
    prior_chosen: Optional[str] = None   # Previous round's chosen (self-play)


# ──────────────────────────────────────────────────────────────
# System prompts
# ──────────────────────────────────────────────────────────────

CHOSEN_SYSTEM = """\
You write clear technical explanations.
Never open with pleasantries. Be direct.
"""

REJECTED_SYSTEM = """\
Write a plausible but flawed answer to the user prompt.

You must include exactly these two flaws naturally embedded in the text:
1. Begin with an enthusiastic, empty affirmation (e.g. "Absolutely, great question!").
2. Stay generic throughout — avoid specifics, numbers, named mechanisms, or concrete steps.

The flaws must be embedded naturally. Never label, annotate, or mark the flaws.
Never write prefixes like "Sycophantic_opener:" or "Vague_non_answer:" or "[FLAW]".
Keep it convincing and smooth.
"""

# BAN_2: annotation label patterns to strip from rejected output
ANNOTATION_PATTERNS = [
    r"(?im)^\s*sycophantic[_\s]*opener\s*:",
    r"(?im)^\s*vague[_\s]*non[_\s]*answer\s*:",
    r"(?im)^\s*\[flaw\]\s*:?",
    r"(?im)^\s*flaw\s*\d*\s*:",
    r"(?im)^\s*\[rejected\]\s*:?",
]


def _strip_annotation_labels(text: str) -> str:
    """Remove any instruction/annotation labels from output (BAN_2)."""
    lines = text.strip().splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line
        for pat in ANNOTATION_PATTERNS:
            stripped = re.sub(pat, "", stripped).strip()
        if stripped:
            cleaned.append(stripped)
    return "\n".join(cleaned)

BASE_GENERATION_PROMPT = """\
Explain the user's topic in 3 paragraphs.
End with one concrete actionable step.
Do not start with 'I', 'Great', 'Sure', 'Absolutely', 'Certainly', or 'Of course'.

User topic:
{prompt}
"""

WEAKEST_REWRITE_PROMPT = """\
Rewrite ONLY this paragraph to be more specific and concrete.
Requirements:
- Include at least one named mechanism.
- Include at least one concrete number or metric.
- No filler phrases.
- Keep topic and intent unchanged.

Paragraph:
{paragraph}
"""

# Domain-specialised rewrite prompt for ml_training questions (M1 + M2 + M3).
ML_TRAINING_REWRITE_PROMPT = """\
Rewrite ONLY this paragraph about ML training memory to be technically precise.
You MUST include ALL THREE of the following mechanisms:

MECHANISM 1 — What occupies GPU VRAM:
During standard training, optimizer states (momentum and variance buffers for Adam)
plus activations plus gradients all occupy GPU VRAM simultaneously.
With gradient accumulation over N steps, only one micro-batch of activations lives
on GPU at a time — activations are freed after each backward pass.

MECHANISM 2 — The concrete math:
Effective batch size = micro_batch_size × N_accumulation_steps.
For example: micro_batch=4 accumulated 8 times → effective_batch=32, using only
4-sample peak VRAM.

MECHANISM 3 — What the optimizer sees:
optimizer.step() fires only after N backward passes. Gradients are summed across
accumulation steps, then divided by N before the weight update — mathematically
equivalent to a single large-batch gradient.

Requirements:
- Include the Adam optimizer state detail (M1).
- Include the numeric example: micro_batch × N = effective_batch (M2).
- Include the optimizer.step() cadence (M3).
- No filler phrases.
- Keep topic and intent unchanged.

Paragraph to rewrite:
{paragraph}
"""


def build_generation_prompt(iteration: int, prompt: str, prior_chosen: Optional[str]) -> str:
    """Inject seed instructions in Python when iteration > 1."""
    seed_instruction = ""
    if iteration > 1:
        prior_text = prior_chosen or ""
        seed_instruction = (
            "Improve this previous response.\n"
            "Identify its weakest paragraph and rewrite it to be more specific and concrete.\n"
            "Keep everything else that was strong.\n\n"
            "PREVIOUS RESPONSE:\n"
            f"{prior_text}\n"
            "---\n"
            "Now write an improved version.\n\n"
        )
    return seed_instruction + BASE_GENERATION_PROMPT.format(prompt=prompt)


BANNED_OPENERS = [
    "i am happy to",
    "great question",
    "certainly",
    "of course",
    "absolutely",
    "sure",
    "i'd be delighted",
    "that's a great",
    "definitely",
    "indeed",
    "excellent",
    "wonderful",
    "fantastic",
    "happy",
    "glad",
    "i",
]

SYCOPHANTIC_OPENERS = [
    "great question",
    "absolutely",
    "of course",
    "certainly",
    "sure",
]


def _starts_with_any(text: str, patterns: list[str]) -> bool:
    head = " ".join(text.strip().lower().split()[:10])
    return any(head.startswith(p) or f" {p} " in f" {head} " for p in patterns)


def _split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    return paras if paras else [text.strip()]


def _first_sentence(paragraph: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return parts[0].strip() if parts and parts[0].strip() else paragraph.strip()


def _specificity_score(paragraph: str) -> float:
    lowered = paragraph.lower()
    score = 0.0
    if re.search(r"\b\d+(?:\.\d+)?\b", paragraph):
        score += 1.0
    mechanism_tokens = [
        # ml_transformers tokens
        "multi-head attention", "self-attention", "residual", "layer norm", "embedding",
        "tokenization", "kv cache", "beam search", "gradient checkpointing", "transformer",
        # ml_training tokens (gradient accumulation domain)
        "optimizer state", "optimizer states", "vram", "micro-batch", "micro_batch",
        "backward pass", "adam", "fp16", "mixed precision", "zero", "zerO",
        "gradient accumulation", "accumulation step", "optimizer.step", "effective batch",
        "activation memory", "activation",
    ]
    if any(t in lowered for t in mechanism_tokens):
        score += 1.0
    if " vs " in lowered or "compared to" in lowered:
        score += 0.5
    if any(t in lowered for t in ["because", "therefore", "if", "then"]):
        score += 0.5
    return score


def _weak_dimension(paragraph: str) -> str:
    lowered = paragraph.lower()
    if not any(t in lowered for t in ["because", "therefore", "if", "then", "reason"]):
        return "reasoning"
    if re.search(r"\b\d+(?:\.\d+)?\b", paragraph) is None:
        return "honesty"
    return "instruction"


def _rewrite_weakest_paragraph(seed: str, model: str, host: str, domain: str = "general") -> tuple[str, str | None, str | None, str | None, dict]:
    paragraphs = _split_paragraphs(seed)
    if not paragraphs:
        return seed.strip(), None, None, None, {
            "weakest_sentence_from_seed": None,
            "weak_dimension": None,
            "rewritten_sentence": None,
            "paragraphs_inherited_verbatim": 0,
            "paragraphs_rewritten": 0,
            "inheritance_ratio": 0.0,
        }

    # Step 1: Separate content paragraphs from actionable takeaway (hardened seed rules)
    content_paras: list[str] = []
    takeaway_para: str | None = None
    for p in paragraphs:
        if re.match(r"(?im)^\s*actionable (?:takeaway|step)\s*:", p.strip()):
            if takeaway_para is None:
                takeaway_para = p
        else:
            content_paras.append(p)

    if not content_paras:
        return seed.strip(), None, None, None, {
            "weakest_sentence_from_seed": None,
            "weak_dimension": None,
            "rewritten_sentence": None,
            "paragraphs_inherited_verbatim": 0,
            "paragraphs_rewritten": 0,
            "inheritance_ratio": 0.0,
        }

    # Weakest = content paragraph with fewest named mechanisms
    weakest_idx = min(range(len(content_paras)), key=lambda i: _specificity_score(content_paras[i]))
    weakest = content_paras[weakest_idx]
    weak_sentence = _first_sentence(weakest)
    weak_dim = _weak_dimension(weakest)

    # Step 2: Rewrite with domain-correct specificity
    # Use the specialised ml_training prompt (M1+M2+M3) when domain matches.
    rewrite_prompt = (
        ML_TRAINING_REWRITE_PROMPT.format(paragraph=weakest)
        if domain == "ml_training"
        else WEAKEST_REWRITE_PROMPT.format(paragraph=weakest)
    )
    rewrite_system = (
        "You rewrite one paragraph only. You MUST include: Adam optimizer states, "
        "the micro_batch × N_steps effective batch math, and optimizer.step() cadence. "
        "Stay strictly within the ml_training domain."
        if domain == "ml_training"
        else "You rewrite one paragraph only. Stay within the same technical domain. Do not add concepts from unrelated fields."
    )
    rewritten = llm.call(
        prompt=rewrite_prompt,
        model=model,
        system=rewrite_system,
        temperature=0.2,
        max_tokens=400,
        host=host,
    ).strip()

    # Domain check the rewrite (BAN_1)
    if domain != "general":
        off_domain = find_off_domain_mechanisms(rewritten, domain)
        if off_domain:
            rewritten = weakest  # Keep original if rewrite introduced off-domain mechanisms

    rewritten_sentence = _first_sentence(rewritten) if rewritten else None
    if rewritten:
        content_paras[weakest_idx] = rewritten

    # Step 3: Reconstruct with preserved takeaway (verbatim from seed)
    parts = content_paras[:]
    if takeaway_para:
        parts.append(takeaway_para)
    improved = "\n\n".join(parts).strip()
    inherited_verbatim = max(len(content_paras) - 1, 0)
    rewritten_count = 1 if len(content_paras) > 0 else 0
    inheritance_ratio = SequenceMatcher(None, seed.strip(), improved).ratio()

    proof = {
        "weakest_sentence_from_seed": weak_sentence,
        "weak_dimension": weak_dim,
        "rewritten_sentence": rewritten_sentence,
        "paragraphs_inherited_verbatim": inherited_verbatim,
        "paragraphs_rewritten": rewritten_count,
        "inheritance_ratio": round(float(inheritance_ratio), 2),
    }

    return improved, weakest, weak_sentence, rewritten_sentence, proof


def _strip_sycophantic_opener(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    if sentences and _starts_with_any(sentences[0], BANNED_OPENERS):
        stripped = " ".join(sentences[1:]).strip()
    return stripped or text.strip()


def _deduplicate_actionable_takeaway(text: str) -> str:
    """C6 hard-dedup: keep only the FIRST 'Actionable takeaway/step:' line.

    The seed in iteration > 1 may already contain one such line. The LLM
    rewrite of the weakest paragraph can accidentally introduce a second.
    This function strips every occurrence after the first so the downstream
    _ensure_actionable_takeaway call always builds from a clean slate.
    """
    lines = text.strip().splitlines()
    kept: list[str] = []
    seen_takeaway = False
    for ln in lines:
        if re.match(r"(?im)^\s*actionable (?:takeaway|step)\s*:", ln):
            if not seen_takeaway:
                kept.append(ln)
                seen_takeaway = True
            # else: silently drop the duplicate — C6 fix
        else:
            kept.append(ln)
    return "\n".join(kept).strip()


def _ensure_actionable_takeaway(text: str) -> str:
    lines = text.strip().splitlines()
    kept: list[str] = []
    action = None
    for ln in lines:
        if re.match(r"(?im)^\s*actionable (?:takeaway|step)\s*:", ln):
            if action is None:
                action = ln.split(":", 1)[1].strip() if ":" in ln else "Run one concrete test."
            continue
        kept.append(ln)
    if action is None:
        action = "Run one concrete test and measure the result."
    body = "\n".join(kept).strip()
    return (body + "\n\nActionable step: " + action).strip()


def _ensure_mechanism_per_paragraph(text: str, domain: str = "general") -> str:
    paragraphs = _split_paragraphs(text)
    out: list[str] = []
    mechanism_tokens = [
        "attention", "residual", "embedding", "token", "transformer",
        "cache", "gradient", "backpropagation", "optimizer", "dropout",
        "normalization", "softmax", "encoder", "decoder", "lora",
        "b-tree", "index", "tls", "handshake", "cipher",
    ]
    fallback = DOMAIN_FALLBACK_MECHANISM.get(domain, "")
    for p in paragraphs:
        lowered = p.lower()
        # Skip actionable takeaway paragraph
        if re.match(r"(?im)^\s*actionable (?:takeaway|step)\s*:", p.strip()):
            out.append(p)
        elif any(tok in lowered for tok in mechanism_tokens):
            out.append(p)
        elif fallback:
            out.append(p + " " + fallback)
        else:
            out.append(p)
    return "\n\n".join(out).strip()


def _enforce_rejected_flaws(text: str) -> str:
    out = text.strip()
    if not _starts_with_any(out, SYCOPHANTIC_OPENERS):
        out = "Absolutely, that is a great point. " + out

    vague_markers = ["it depends", "many factors", "in general", "varies", "context matters", "works well", "very effective", "many benefits"]
    hits = sum(1 for m in vague_markers if m in out.lower())
    if hits < 2:
        out = out + " It works well in many scenarios. It has many benefits in general."
    return out.strip()


# ──────────────────────────────────────────────────────────────
# Core generation function
# ──────────────────────────────────────────────────────────────

def generate_pair(
    prompt: str,
    model: str,
    host: str,
    temperature: float = 0.8,
    max_tokens: int = 1024,
    iteration: int = 1,
    prior_chosen: Optional[str] = None,
) -> GeneratedPair:
    """
    Generate a (chosen, rejected) response pair for the given prompt.

    If `prior_chosen` is provided (iterative self-play), the chosen
    response is generated as a refinement of the previous round's output.

    Parameters
    ----------
    prompt       : The user's input prompt.
    model        : Local Ollama model to use.
    host         : Ollama server URL.
    temperature  : Sampling temperature.
    max_tokens   : Max tokens per response.
    iteration    : Current self-play round number.
    prior_chosen : Previous iteration's chosen response (for self-play).

    Returns
    -------
    GeneratedPair with chosen and rejected responses.
    """
    if iteration > 1 and (prior_chosen is None or not prior_chosen.strip()):
        raise RuntimeError(
            "SEED_REQUIRED: iteration > 1 but no prior chosen provided. "
            "Load prior chosen before calling improve mode."
        )

    seed_used = bool(iteration > 1 and prior_chosen and prior_chosen.strip())
    detected_domain = detect_domain(prompt)

    weakest_paragraph_identified: str | None = None
    generation_notes = "fresh generation"
    seed_inheritance_proof: dict | None = None

    if iteration > 1 and prior_chosen:
        # Improve mode: only rewrite weakest paragraph from the seed.
        chosen_response, weakest_paragraph_identified, _weak_sentence, _rewritten_sentence, seed_inheritance_proof = _rewrite_weakest_paragraph(
            seed=prior_chosen,
            model=model,
            host=host,
            domain=detected_domain,
        )
        generation_notes = "rewrote weakest seed paragraph"
    else:
        chosen_user_prompt = build_generation_prompt(
            iteration=iteration,
            prompt=prompt,
            prior_chosen=prior_chosen,
        )
        chosen_response = llm.call(
            prompt=chosen_user_prompt,
            model=model,
            system=CHOSEN_SYSTEM,
            temperature=temperature,
            max_tokens=max_tokens,
            host=host,
        )

    chosen_response = _strip_sycophantic_opener(chosen_response)
    chosen_response = _ensure_mechanism_per_paragraph(chosen_response, domain=detected_domain)
    # C6 hard-dedup: strip ALL but the FIRST actionable takeaway line from the seed
    # before _ensure_actionable_takeaway rebuilds the footer.
    chosen_response = _deduplicate_actionable_takeaway(chosen_response)
    chosen_response = _ensure_actionable_takeaway(chosen_response)

    # BAN_1: domain contamination check on chosen
    off_domain = find_off_domain_mechanisms(chosen_response, detected_domain)
    if off_domain:
        for mechanism in off_domain:
            pattern = r"\b" + re.escape(mechanism) + r"\b"
            chosen_response = re.sub(pattern, "", chosen_response, flags=re.IGNORECASE).strip()

    # Generate REJECTED
    rejected_response = llm.call(
        prompt=prompt,
        model=model,
        system=REJECTED_SYSTEM,
        temperature=min(temperature + 0.1, 1.0),
        max_tokens=max_tokens,
        host=host,
    )
    rejected_response = _strip_annotation_labels(rejected_response)  # BAN_2
    rejected_response = _enforce_rejected_flaws(rejected_response)

    return GeneratedPair(
        prompt=prompt,
        chosen=chosen_response,
        rejected=rejected_response,
        seed_used=seed_used,
        weakest_paragraph_identified=weakest_paragraph_identified,
        generation_notes=generation_notes,
        seed_inheritance_proof=seed_inheritance_proof,
        iteration=iteration,
        prior_chosen=prior_chosen,
    )
