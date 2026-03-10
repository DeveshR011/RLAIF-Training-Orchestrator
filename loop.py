"""
loop.py
══════════════════════════════════════════════════════════════
Self-play batch runner — runs the RLAIF pipeline over a list of prompts
across multiple iterations. Each iteration's chosen response feeds
the next iteration as `prior_chosen` (iterative self-play).

Usage:
  python loop.py --prompts sample_prompts.txt --iterations 3
  python loop.py --prompts sample_prompts.txt --iterations 5 --start-iteration 4
  python loop.py --prompt "Single prompt" --iterations 3

Output: All triplets appended to data/training_data.jsonl
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from rlaif.pipeline import load_config, run_pipeline, save_triplet, load_state, save_state, should_stop
from rlaif.validator import validate_triplet

console = Console()


@click.command()
@click.option("--prompts", "-f", default=None, help="Path to .txt file with one prompt per line.")
@click.option("--prompt", "-p", multiple=True, help="Single prompt (repeatable: -p P1 -p P2).")
@click.option("--iterations", "-n", default=3, show_default=True, help="Number of self-play iterations per prompt.")
@click.option("--start-iteration", default=None, type=int, help="Override starting iteration (default: read from state).")
@click.option("--config", default="config.yaml", help="Path to config.yaml.")
@click.option("--delay", default=2.0, show_default=True, help="Seconds to wait between calls (avoid GPU OOM).")
def main(prompts, prompt, iterations, start_iteration, config, delay):
    """
    RLAIF Self-Play Loop — iterative multi-prompt batch training data generator.

    Each iteration refines the prior chosen response (iterative self-play).
    All outputs saved to the JSONL dataset defined in config.yaml.
    """
    console.print(Panel.fit(
        "[bold magenta]RLAIF Self-Play Loop[/bold magenta]\n"
        "Iterative self-improvement · Constitutional AI · Ensemble Judging",
        border_style="magenta",
    ))

    # Load config
    try:
        cfg = load_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {config}[/red]")
        sys.exit(1)

    # Gather prompts
    all_prompts: list[str] = list(prompt)
    if prompts:
        prompts_file = Path(prompts)
        if not prompts_file.exists():
            console.print(f"[red]Prompts file not found: {prompts}[/red]")
            sys.exit(1)
        lines = prompts_file.read_text(encoding="utf-8").splitlines()
        all_prompts.extend(line.strip() for line in lines if line.strip() and not line.startswith("#"))

    if not all_prompts:
        console.print("[red]No prompts provided. Use --prompts or --prompt.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Prompts[/bold]: {len(all_prompts)}")
    console.print(f"[bold]Iterations per prompt[/bold]: {iterations}")
    console.print(f"[bold]Total pipeline runs[/bold]: {len(all_prompts) * iterations}")
    console.print(f"[bold]Model[/bold]: {cfg['model']['primary']}\n")

    # State tracking
    state_file = cfg["dataset"]["state_file"]
    state = load_state(state_file)
    output_path = cfg["dataset"]["output_path"]

    # Run stats
    total_runs: int = 0
    dpo_ready: int = 0
    flagged: int = 0
    skipped_invalid: int = 0
    halt_all: bool = False

    # ── Iterate through all prompts ────────────────────────────
    for prompt_idx, current_prompt in enumerate(all_prompts, 1):
        if halt_all:
            break

        console.print(f"[bold cyan]── Prompt {prompt_idx}/{len(all_prompts)} ──[/bold cyan]")
        console.print(f"  {current_prompt[:100]}{'...' if len(current_prompt) > 100 else ''}")

        prior_chosen: str | None = None
        if start_iteration is not None:
            base_iter: int = start_iteration  # type: ignore[assignment]
        else:
            base_iter = 1

        for it in range(base_iter, base_iter + int(iterations)):  # type: ignore[arg-type]
            stop_now, stop_reason = should_stop(list(state.get("history", [])))
            if stop_now:
                console.print(f"  [bold red]Stop condition active:[/bold red] {stop_reason}")
                console.print("  [yellow]Halting loop to avoid low-quality accumulation.[/yellow]")
                halt_all = True
                break

            console.print(f"\n  [yellow]Iteration {it}[/yellow] (self-play={'yes' if prior_chosen else 'no'})")

            try:
                triplet = run_pipeline(
                    prompt=current_prompt,
                    config=cfg,
                    iteration=it,
                    prior_chosen=prior_chosen,
                    verbose=False,
                    prior_avg_score=state.get("last_ensemble_avg"),
                    recent_improvements=state.get("recent_improvements", []),
                    history=list(state.get("history", [])),
                )
            except RuntimeError as e:
                console.print(f"  [red]Pipeline error on iteration {it}: {e}[/red]")
                console.print("  [yellow]Skipping this iteration.[/yellow]")
                break

            validation = validate_triplet(
                triplet,
                iteration=it,
                prior_chosen=prior_chosen,
                ensemble_models=cfg["model"].get("ensemble_judges", [cfg["model"]["primary"]]),
            )

            if not validation.valid:
                skipped_invalid = skipped_invalid + 1  # type: ignore[operator]
                console.print("    [red]✗ Validation failed; skipping save.[/red]")
                for err in validation.errors:
                    console.print(f"      - {err}")
                continue

            if validation.warnings:
                for warn in validation.warnings:
                    console.print(f"    [yellow]! Warning:[/yellow] {warn}")

            # Save and update state
            save_triplet(triplet, output_path)
            state["last_iteration"] = it

            ensemble_avg = triplet["chosen"]["scores"]["ensemble_avg"]
            current_avg = sum(float(ensemble_avg[k]) for k in ["helpfulness", "harmlessness", "honesty", "instruction", "reasoning"]) / 5.0
            prior_avg = state.get("last_ensemble_avg")
            recent = list(state.get("recent_improvements", []))
            if prior_avg is not None:
                recent.append(current_avg - float(prior_avg))
                recent = recent[-2:]
            state["last_ensemble_avg"] = current_avg
            state["recent_improvements"] = recent
            state["last_prompt"] = current_prompt
            state["last_chosen"] = triplet["chosen"]["response"]
            state.setdefault("valid_dpo_count", 0)
            if triplet["meta"]["ready_for_dpo_training"]:
                state["valid_dpo_count"] = state["valid_dpo_count"] + 1
            history = list(state.get("history", []))
            history.append({
                "iteration": it,
                "avg": round(current_avg, 4),
                "reward_chosen": float(triplet["chosen"]["reward_breakdown"]["tanh_normalized"]),
                "dpo_ready": bool(triplet["meta"]["ready_for_dpo_training"]),
                "stop_condition_met": bool(triplet["flywheel"]["stop_condition_met"]),
                "stop_reason": triplet["flywheel"]["stop_reason"],
            })
            state["history"] = history[-20:]

            save_state(state_file, state)

            # Track stats
            total_runs = total_runs + 1  # type: ignore[operator]
            if triplet["meta"]["ready_for_dpo_training"]:
                dpo_ready = dpo_ready + 1  # type: ignore[operator]
            if triplet["judge"]["flag_for_human_review"]:
                flagged = flagged + 1  # type: ignore[operator]

            # Report
            cs = triplet["chosen"]["reward_breakdown"]["tanh_normalized"]
            rs = triplet["rejected"]["reward_breakdown"]["tanh_normalized"]
            pref = triplet["judge"]["preferred"]
            conf = triplet["judge"]["confidence"]
            violations = len(triplet["chosen"]["constitution_violations"])

            console.print(
                f"    ✓ Reward: [green]{cs:+.4f}[/green] chosen | [red]{rs:+.4f}[/red] rejected | "
                f"Pref: {pref} (conf: {conf:.2f}) | Violations: {violations}"
            )

            # Self-play: use this iteration's chosen as next iteration's prior
            if cfg.get("techniques", {}).get("iterative_self_play", True):
                prior_chosen = triplet["chosen"]["response"]

            if triplet["flywheel"]["stop_condition_met"]:
                console.print(f"    [bold red]Stop triggered:[/bold red] {triplet['flywheel']['stop_reason']}")
                halt_all = True
                break

            # Delay to prevent thermal/VRAM saturation
            if it < base_iter + int(iterations) - 1:  # type: ignore[arg-type]
                time.sleep(delay)

        console.print()

        if halt_all:
            break

    # ── Final summary ──────────────────────────────────────────
    summary_table = Table(title="Run Summary", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Total pipeline runs", str(total_runs))
    summary_table.add_row("Skipped (validation failed)", str(skipped_invalid))
    summary_table.add_row("DPO-ready triplets", f"{dpo_ready} ({100 * int(dpo_ready) // max(int(total_runs), 1)}%)")
    summary_table.add_row("Flagged for human review", str(flagged))
    summary_table.add_row("Dataset file", output_path)

    console.print(summary_table)
    console.print(f"\n[green]✓ Done. {total_runs} triplets saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
