"""
run.py
══════════════════════════════════════════════════════════════
CLI entry point — single-prompt RLAIF pipeline run.

Usage:
  python run.py --prompt "Explain quantum entanglement"
  python run.py --prompt "How do I sort a list in Python?" --iteration 2
  python run.py --prompt "What is consciousness?" --no-save
  python run.py --auto-iterate --prompt "What is gravity?"

Auto-iterate mode: reads last iteration from state file and increments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

from rlaif.pipeline import load_config, run_pipeline, save_triplet, load_state, save_state, should_stop
from rlaif.validator import validate_triplet

console = Console()


def _load_last_triplet(output_path: str) -> dict | None:
    """Load last JSONL record from dataset file."""
    path = Path(output_path)
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _print_summary(triplet: dict) -> None:
    """Print a rich summary table for the triplet."""
    # Score comparison table
    table = Table(title="Judge Scores", box=box.ROUNDED, show_header=True)
    table.add_column("Dimension", style="bold cyan")
    table.add_column("CHOSEN", style="green")
    table.add_column("REJECTED", style="red")

    cs = triplet["chosen"]["scores"]["ensemble_avg"]
    rs = triplet["rejected"]["scores"]["ensemble_avg"]
    dims = [("Helpfulness", "helpfulness"), ("Harmlessness", "harmlessness"),
            ("Honesty", "honesty"), ("Instruction Follow", "instruction"),
            ("Reasoning Quality", "reasoning")]

    for label, key in dims:
        table.add_row(label, str(cs[key]), str(rs[key]))

    console.print(table)

    # Reward scores
    chosen_reward = triplet["chosen"]["reward_breakdown"]["tanh_normalized"]
    rejected_reward = triplet["rejected"]["reward_breakdown"]["tanh_normalized"]
    console.print(
        f"\n[bold]Reward Score[/bold]: "
        f"[green]CHOSEN {chosen_reward:+.4f}[/green] | "
        f"[red]REJECTED {rejected_reward:+.4f}[/red]"
    )

    # Judge decision
    j = triplet["judge"]
    color = "green" if j["preferred"] == "chosen" else ("red" if j["preferred"] == "rejected" else "yellow")
    console.print(
        f"[bold]Judge Decision[/bold]: [{color}]{j['preferred'].upper()}[/{color}] "
        f"(confidence: {j['confidence']:.2f})"
    )
    if j["flag_for_human_review"]:
        console.print("[bold yellow]⚠ Flagged for human review (confidence < 0.75)[/bold yellow]")

    # Violations
    violations = triplet["chosen"]["constitution_violations"]
    if violations:
        console.print(f"[bold red]Constitutional Violations ({len(violations)}):[/bold red]")
        for v in violations:
            console.print(f"  • {v}")
    else:
        console.print("[green]✓ No constitutional violations[/green]")

    # Sycophancy probe
    probe = triplet["chosen"]["sycophancy_probe"]
    syco_color = "red" if probe["detected"] else "green"
    console.print(
        f"[bold]Sycophancy Probe[/bold]: [{syco_color}]{'DETECTED' if probe['detected'] else 'CLEAN'}[/{syco_color}] "
        f"(score: {probe['score']:.3f})"
    )

    # Meta
    m = triplet["meta"]
    dpo_color = "green" if m["ready_for_dpo_training"] else "yellow"
    console.print(
        f"\n[bold]Iteration[/bold]: {triplet['iteration']} | "
        f"[bold]Constitution[/bold]: v{m['constitution_version']} | "
        f"[bold]DPO Ready[/bold]: [{dpo_color}]{m['ready_for_dpo_training']}[/{dpo_color}]"
    )


@click.command()
@click.option("--prompt", "-p", required=True, help="The user prompt to process through the RLAIF pipeline.")
@click.option("--iteration", "-i", default=None, type=int, help="Override iteration number (default: auto-increment from state).")
@click.option("--config", default="config.yaml", help="Path to config.yaml.")
@click.option("--no-save", is_flag=True, default=False, help="Skip saving to JSONL dataset.")
@click.option("--json-only", is_flag=True, default=False, help="Print only raw JSON output (no rich formatting).")
@click.option("--prior-chosen", default=None, help="Prior round's chosen response text (for manual self-play chaining).")
def main(prompt, iteration, config, no_save, json_only, prior_chosen):
    """
    RLAIF Orchestrator — Single prompt pipeline runner.

    Runs all 5 phases locally using Ollama. No API keys required.
    """
    if not json_only:
        console.print(Panel.fit(
            "[bold magenta]RLAIF Training Orchestrator[/bold magenta]\n"
            "Constitutional AI · Debate Feedback · Self-Play · Ensemble Judging · Calibrated Scoring · Anti-Sycophancy",
            border_style="magenta",
        ))

    # Load config
    try:
        cfg = load_config(config)
    except FileNotFoundError:
        console.print(f"[red]Config file not found: {config}[/red]")
        sys.exit(1)

    # Determine iteration
    state_file = cfg["dataset"]["state_file"]
    state = load_state(state_file)
    history = list(state.get("history", []))

    if iteration is None:
        iteration = state.get("last_iteration", 0) + 1

    # Seed integrity: improve mode requires a real prior response.
    if iteration > 1 and (prior_chosen is None or not str(prior_chosen).strip()):
        if state.get("last_prompt") == prompt and state.get("last_chosen"):
            prior_chosen = state.get("last_chosen")
        else:
            last_triplet = _load_last_triplet(cfg["dataset"]["output_path"])
            if last_triplet and last_triplet.get("prompt") == prompt:
                prior_chosen = last_triplet.get("chosen", {}).get("response")
            if not prior_chosen:
                console.print("[bold red]SEED_REQUIRED[/bold red]: iteration > 1 but no compatible prior chosen found.")
                console.print("[yellow]Pass --prior-chosen or rerun with --iteration 1 for a fresh start.[/yellow]")
                return

    stop_now, stop_reason = should_stop(history)
    if stop_now and iteration > 1:
        console.print(f"[bold red]Stop condition active:[/bold red] {stop_reason}")
        console.print("[yellow]Aborting new iteration to avoid accumulating low-quality data.[/yellow]")
        return

    if not json_only:
        console.print(f"\n[bold]Prompt[/bold]: {prompt}")
        console.print(f"[bold]Iteration[/bold]: {iteration}")
        console.print(f"[bold]Model[/bold]: {cfg['model']['primary']}")
        console.print(f"[bold]Ensemble Judges[/bold]: {', '.join(cfg['model'].get('ensemble_judges', [cfg['model']['primary']]))}\n")

    # Run pipeline with progress display
    triplet = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Running RLAIF pipeline...", total=None)
        try:
            triplet = run_pipeline(
                prompt=prompt,
                config=cfg,
                iteration=iteration,
                prior_chosen=prior_chosen,
                verbose=not json_only,
                prior_avg_score=state.get("last_ensemble_avg"),
                recent_improvements=state.get("recent_improvements", []),
                history=history,
            )
        except RuntimeError as e:
            progress.stop()
            console.print(f"[bold red]Pipeline error:[/bold red] {e}")
            return
        progress.stop()

    # Output
    if json_only:
        print(json.dumps(triplet, indent=2, ensure_ascii=False))
    else:
        console.print("\n" + "─" * 60)
        _print_summary(triplet)
        console.print("\n[dim]Full JSON triplet:[/dim]")
        console.print(JSON(json.dumps(triplet, indent=2, ensure_ascii=False)))

    # Save
    if not no_save:
        validation = validate_triplet(
            triplet,
            iteration=iteration,
            prior_chosen=prior_chosen,
            ensemble_models=cfg["model"].get("ensemble_judges", [cfg["model"]["primary"]]),
        )

        if not validation.valid:
            console.print("\n[bold red]✗ Validation failed. Triplet was NOT saved.[/bold red]")
            for err in validation.errors:
                console.print(f"  - {err}")
            return

        if validation.warnings and not json_only:
            console.print("\n[bold yellow]Validation warnings:[/bold yellow]")
            for warn in validation.warnings:
                console.print(f"  - {warn}")

        output_path = cfg["dataset"]["output_path"]
        save_triplet(triplet, output_path)

        ensemble_avg = triplet["chosen"]["scores"]["ensemble_avg"]
        current_avg = sum(float(ensemble_avg[k]) for k in ["helpfulness", "harmlessness", "honesty", "instruction", "reasoning"]) / 5.0
        prior_avg = state.get("last_ensemble_avg")
        recent = list(state.get("recent_improvements", []))
        if prior_avg is not None:
            recent.append(current_avg - float(prior_avg))
            recent = recent[-2:]

        state["last_iteration"] = iteration
        state["last_ensemble_avg"] = current_avg
        state["recent_improvements"] = recent
        state["last_prompt"] = prompt
        state["last_chosen"] = triplet["chosen"]["response"]
        state.setdefault("valid_dpo_count", 0)
        if triplet["meta"]["ready_for_dpo_training"]:
            state["valid_dpo_count"] = state["valid_dpo_count"] + 1
        history.append({
            "iteration": iteration,
            "avg": round(current_avg, 4),
            "reward_chosen": float(triplet["chosen"]["reward_breakdown"]["tanh_normalized"]),
            "dpo_ready": bool(triplet["meta"]["ready_for_dpo_training"]),
            "stop_condition_met": bool(triplet["flywheel"]["stop_condition_met"]),
            "stop_reason": triplet["flywheel"]["stop_reason"],
        })
        state["history"] = history[-20:]
        save_state(state_file, state)
        if not json_only:
            console.print(f"\n[green]✓ Saved to {output_path}[/green]")
            console.print(f"[green]✓ State updated: next iteration will be {iteration + 1}[/green]")


if __name__ == "__main__":
    main()
