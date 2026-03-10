"""
rlaif/llm.py
──────────────────────────────────────────────────────────────
Local LLM client wrapping Ollama.
Handles:
  - Single inference calls
  - Sequential multi-model calls (ensemble mode, 6GB VRAM safe)
  - Automatic retry on timeout
  - VRAM-safe: models are loaded on-demand by Ollama and
    do NOT need to stay resident between calls.
"""
from __future__ import annotations

import time
from typing import Optional
import ollama


def call(
    prompt: str,
    model: str,
    system: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    host: str = "http://localhost:11434",
    retries: int = 2,
    timeout: int = 120,
) -> str:
    """
    Send a single prompt to a local Ollama model and return the response text.

    Parameters
    ----------
    prompt      : User-facing message.
    model       : Ollama model tag, e.g. 'mistral:7b-instruct-q4_K_M'.
    system      : Optional system prompt.
    temperature : Sampling temperature (0.0 = deterministic, 1.0 = creative).
    max_tokens  : Maximum tokens to generate.
    host        : Ollama server URL.
    retries     : Number of retry attempts on failure.
    timeout     : Per-call timeout in seconds.

    Returns
    -------
    str : The model's response content (stripped).
    """
    client = ollama.Client(host=host)

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
    }

    last_exc: Exception = RuntimeError("No attempts made.")
    for attempt in range(retries + 1):
        try:
            resp = client.chat(
                model=model,
                messages=messages,
                options=options,
            )
            return resp["message"]["content"].strip()
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(
        f"[LLM] Failed after {retries + 1} attempts on model '{model}': {last_exc}"
    ) from last_exc


def ensemble_call(
    prompt: str,
    models: list[str],
    system: str = "",
    temperatures: Optional[list[float]] = None,
    max_tokens: int = 1024,
    host: str = "http://localhost:11434",
) -> list[str]:
    """
    Run inference sequentially across multiple models (VRAM-safe).
    Returns a list of response strings, one per model.

    Each model is fully unloaded by Ollama between calls because
    Ollama evicts the previous model from VRAM before loading the next.
    This keeps peak VRAM ≤ max(single model size).

    Parameters
    ----------
    models       : List of Ollama model tags.
    temperatures : Per-model temperatures; defaults to 0.4 for all.
    """
    if temperatures is None:
        temperatures = [0.4] * len(models)
    if len(temperatures) != len(models):
        raise ValueError("temperatures list must match models list length.")

    responses = []
    for model, temp in zip(models, temperatures):
        resp = call(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temp,
            max_tokens=max_tokens,
            host=host,
        )
        responses.append(resp)
    return responses
