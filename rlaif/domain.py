"""Domain detection and contamination checking (BAN_1)."""
from __future__ import annotations

import re


# ── Domain keyword mapping ────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "ml_transformers": [
        "transformer", "llm", "large language model", "neural network",
        "attention mechanism", "bert", "gpt", "tokeniz", "embedding",
        "self-attention", "multi-head", "positional encoding",
    ],
    "ml_training": [
        "backpropagation", "gradient", "overfit", "fine-tun",
        "learning rate", "batch size", "epoch", "optimizer",
        "loss function", "rlhf", "dpo", "alignment", "lora",
        "gradient accumulation", "gradient checkpointing",
        "vanishing gradient", "residual connection",
    ],
    "ml_general": [
        "supervised", "unsupervised", "machine learning", "deep learning",
        "classification", "regression", "clustering",
    ],
    "software_engineering": [
        "deploy", "kubernetes", "docker", "api", "database", "index",
        "postgresql", "fastapi", "monorepo", "polyrepo",
    ],
    "security": [
        "tls", "ssl", "encryption", "certificate", "handshake",
    ],
}

# ── Valid mechanisms per domain ───────────────────────────────
VALID_MECHANISMS: dict[str, list[str]] = {
    "ml_transformers": [
        "multi-head attention", "positional encoding", "residual connections",
        "layer normalization", "softmax", "feed-forward sublayer",
        "token embeddings", "self-attention", "cross-attention",
        "causal masking", "encoder-decoder", "kv cache",
        "scaled dot-product", "query key value",
    ],
    "ml_training": [
        "backpropagation", "gradient descent", "stochastic gradient descent",
        "adam optimizer", "learning rate schedule", "dropout",
        "gradient accumulation", "gradient checkpointing",
        "weight decay", "batch normalization", "residual connections",
        "vanishing gradient", "chain rule", "lora", "full fine-tuning",
        # Iteration 2 additions — gradient accumulation memory mechanisms
        "optimizer state", "optimizer states", "vram", "gpu vram",
        "micro-batch", "micro_batch", "micro batch",
        "backward pass", "forward pass",
        "fp16", "mixed precision", "zero optimizer", "zero redundancy optimizer",
        "effective batch size", "accumulation steps", "gradient accumulation steps",
        "optimizer.step", "weight update", "activation memory",
        "adam", "momentum buffer", "variance buffer",
    ],
    "ml_general": [
        "decision boundary", "feature space", "loss function",
        "cross-validation", "bias-variance tradeoff", "regularization",
        "supervised learning", "unsupervised learning",
    ],
    "software_engineering": [
        "load balancer", "reverse proxy", "container orchestration",
        "connection pooling", "index scan", "b-tree", "hash index",
    ],
    "security": [
        "tls handshake", "certificate authority", "cipher suite",
        "key exchange", "diffie-hellman", "aes", "public key",
    ],
}

# ── Off-domain concepts (universal ban list) ──────────────────
UNIVERSAL_BANNED = [
    "pomodoro", "gtd", "getting things done", "okr",
    "objectives and key results", "agile methodology",
    "scrum framework", "kanban board", "sprint planning",
    "standup meeting", "retrospective", "waterfall methodology",
    "lean six sigma", "swot analysis", "balanced scorecard",
    "time blocking", "eisenhower matrix",
]

# Domain-scoped fallback mechanism sentences
DOMAIN_FALLBACK_MECHANISM: dict[str, str] = {
    "ml_transformers": "This leverages the self-attention mechanism.",
    "ml_training": "This uses gradient-based optimization.",
    "ml_general": "This applies cross-validation for evaluation.",
    "software_engineering": "This uses connection pooling for efficiency.",
    "security": "This relies on the TLS handshake protocol.",
    "general": "",
}


def detect_domain(prompt: str) -> str:
    """Detect the technical domain from the user prompt."""
    lowered = prompt.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lowered)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "general"
    return max(scores, key=lambda k: scores[k])


def find_off_domain_mechanisms(text: str, domain: str) -> list[str]:
    """Find any off-domain (banned) mechanisms in the text."""
    lowered = text.lower()
    hits: list[str] = []
    for mechanism in UNIVERSAL_BANNED:
        pattern = r"\b" + re.escape(mechanism) + r"\b"
        if re.search(pattern, lowered):
            hits.append(mechanism)
    return list(set(hits))


def is_domain_clean(text: str, domain: str) -> bool:
    """Return True if no off-domain mechanisms are found."""
    return len(find_off_domain_mechanisms(text, domain)) == 0
