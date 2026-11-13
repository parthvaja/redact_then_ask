"""Command-line interface for Redact-Then-Ask.

Built with Typer.  Entry point: ``rta`` (see ``pyproject.toml``).

Available commands::

    rta detect FILE [--format json|table]
    rta redact FILE [--mode mask|placeholder|fake] [--out PATH]
    rta chat   FILE
    rta eval   --dataset DATASET --split SPLIT [--compare presidio] [--system NAME]
    rta classify train|eval
    rta data   fetch|convert
    rta labelstudio export|import

Stages are imported lazily so the CLI loads fast even when heavy
dependencies (spaCy, transformers) are not yet installed.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler

# Top-level Typer app
app = typer.Typer(
    name="rta",
    help="Redact-Then-Ask: privacy-first PII detection and local LLM chat.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_time=False)],
    )


# ---------------------------------------------------------------------------
# Enums for CLI options
# ---------------------------------------------------------------------------


class RedactMode(StrEnum):
    mask = "mask"
    placeholder = "placeholder"
    fake = "fake"


class OutputFormat(StrEnum):
    json = "json"
    table = "table"


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


@app.command()
def detect(
    file: Annotated[Path, typer.Argument(help="Input document (PDF/DOCX/TXT)")],
    format: Annotated[OutputFormat, typer.Option("--format", "-f")] = OutputFormat.table,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run PII detection on FILE and print the detected spans."""
    _setup_logging(verbose)
    _require_file(file)

    from rta.pipeline import Pipeline

    result = Pipeline().run(file)
    spans = result.spans_merged

    if format == OutputFormat.json:
        import json

        console.print(json.dumps([s.to_dict() for s in spans], indent=2, ensure_ascii=False))
    else:
        from rich.table import Table

        tbl = Table(title=f"Detected PII — {file.name}", show_lines=False)
        tbl.add_column("Start", style="dim", no_wrap=True)
        tbl.add_column("End", style="dim", no_wrap=True)
        tbl.add_column("Label", style="bold cyan")
        tbl.add_column("Text", style="yellow")
        tbl.add_column("Score", justify="right")
        tbl.add_column("Detector", style="dim")
        tbl.add_column("Val.", justify="center")
        for s in spans:
            tbl.add_row(
                str(s.start),
                str(s.end),
                s.label,
                repr(s.text[:40]),
                f"{s.score:.2f}",
                s.detector,
                "✓" if s.validated else "",
            )
        console.print(tbl)
        console.print(f"[dim]Total: {len(spans)} span(s)[/dim]")


# ---------------------------------------------------------------------------
# redact
# ---------------------------------------------------------------------------


@app.command()
def redact(
    file: Annotated[Path, typer.Argument(help="Input document")],
    mode: Annotated[RedactMode, typer.Option("--mode", "-m")] = RedactMode.placeholder,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Redact PII in FILE and write the result."""
    _setup_logging(verbose)
    _require_file(file)

    from rta.config import get_config
    from rta.pipeline import Pipeline

    cfg = get_config()
    # Override mode from CLI
    object.__setattr__(cfg.redact, "mode", mode.value)

    result = Pipeline(cfg).run(file)
    if result.redaction is None:
        console.print("[red]Redaction stage did not produce output.[/red]")
        raise typer.Exit(1)

    text = result.redaction.text
    if out is not None:
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]Redacted text written to {out}[/green]")
    else:
        console.print(text)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command()
def chat(
    file: Annotated[Path, typer.Argument(help="Input document")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Redact FILE then start an interactive Q&A session over the redacted text."""
    _setup_logging(verbose)
    _require_file(file)

    from rta.pipeline import Pipeline
    from rta.rag import answer, index

    result = Pipeline().run(file)
    if result.redaction is None:
        console.print("[red]Redaction stage did not produce output.[/red]")
        raise typer.Exit(1)

    store = index(result.redaction.text)
    console.print("[bold green]Document indexed. Type your question (Ctrl-C to quit).[/bold green]")

    while True:
        try:
            question = typer.prompt("You")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        response = answer(store, question)
        console.print(f"[bold cyan]Assistant:[/bold cyan] {response.text}")
        if response.sources:
            console.print(f"  [dim]Sources: {', '.join(str(s) for s in response.sources)}[/dim]")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@app.command()
def eval(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="ai4privacy | tab | handlabel")],
    split: Annotated[str, typer.Option("--split", "-s", help="train | test | val")] = "test",
    compare: Annotated[str | None, typer.Option("--compare", help="presidio")] = None,
    system: Annotated[
        str, typer.Option("--system", help="Label for our system in the report")
    ] = "ours",
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the evaluation harness and write reports to reports/tables/."""
    _setup_logging(verbose)

    from rta.eval import run_evaluation

    run_evaluation(dataset=dataset, split=split, compare=compare, system_name=system)


# ---------------------------------------------------------------------------
# classify sub-app
# ---------------------------------------------------------------------------

classify_app = typer.Typer(help="Build and evaluate the document classifier.")
app.add_typer(classify_app, name="classify")


@classify_app.command("train")
def classify_train(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Train the TF-IDF + LR document classifier."""
    _setup_logging(verbose)
    from rta.classify import train

    train()


@classify_app.command("eval")
def classify_eval(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Evaluate the trained document classifier."""
    _setup_logging(verbose)
    from rta.classify import evaluate

    evaluate()


# ---------------------------------------------------------------------------
# data sub-app
# ---------------------------------------------------------------------------

data_app = typer.Typer(help="Download and convert datasets.")
app.add_typer(data_app, name="data")


@data_app.command("fetch")
def data_fetch(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Download all public datasets."""
    _setup_logging(verbose)
    from rta.data.ai4privacy import fetch as ai4_fetch
    from rta.data.tab import fetch as tab_fetch

    ai4_fetch()
    tab_fetch()


@data_app.command("convert")
def data_convert(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Convert raw datasets to the shared JSONL span format."""
    _setup_logging(verbose)
    from rta.data.ai4privacy import convert as ai4_convert
    from rta.data.tab import convert as tab_convert

    ai4_convert()
    tab_convert()


# ---------------------------------------------------------------------------
# labelstudio sub-app
# ---------------------------------------------------------------------------

ls_app = typer.Typer(help="Round-trip pre-annotations with Label Studio.")
app.add_typer(ls_app, name="labelstudio")


@ls_app.command("export")
def ls_export(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Export pipeline pre-annotations to Label Studio JSON."""
    _setup_logging(verbose)
    from rta.data.labelstudio import export

    export()


@ls_app.command("import")
def ls_import(
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Import corrected annotations from Label Studio and convert to GoldDoc JSONL."""
    _setup_logging(verbose)
    from rta.data.labelstudio import import_annotations

    import_annotations()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _require_file(path: Path) -> None:
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)
    if not path.is_file():
        console.print(f"[red]Not a file: {path}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
