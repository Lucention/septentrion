"""Command-line entry points for the PoC."""

from __future__ import annotations

import typer

from septentrion.config import Config
from septentrion.pipelines.skeleton import run_skeleton

app = typer.Typer(help="Septentrion Aperture PoC commands.")


@app.callback()
def _callback() -> None:  # pragma: no cover
    """Septentrion Aperture PoC CLI."""


@app.command("run-skeleton")
def run_skeleton_command(
    out_dir: str = typer.Option("outputs", help="Directory for results."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Run the synthetic walking skeleton end to end."""
    cfg = Config(out_dir=out_dir, seed=seed)
    result = run_skeleton(cfg)
    typer.echo(
        f"detections={result.n_detections} matched={result.n_matched} "
        f"dark={result.n_dark} separation_auc={result.separation_auc:.3f}"
    )
