"""
ioc-miner CLI — extract IOCs from threat intel sources.

Examples:
  ioc-miner extract report.pdf --format stix
  ioc-miner extract report.html --format csv --no-benign
  cat report.txt | ioc-miner extract - --format stdout
  ioc-miner extract https://threatreport.example/blog --format misp
  ioc-miner extract channel.txt --source telegram --limit 500
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="ioc-miner",
    help="Extract and classify IOCs from threat intelligence sources.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


class OutputFormat(str, Enum):
    stix = "stix"
    csv = "csv"
    misp = "misp"
    stdout = "stdout"


class SourceType(str, Enum):
    auto = "auto"
    pdf = "pdf"
    html = "html"
    text = "text"
    telegram = "telegram"
    twitter = "twitter"


def _detect_source_type(source: str) -> SourceType:
    if source == "-":
        return SourceType.text
    if source.startswith(("http://", "https://")):
        return SourceType.html
    if source.startswith("@") or "twitter.com" in source or "x.com" in source:
        return SourceType.twitter
    if source.startswith("t.me/"):
        return SourceType.telegram
    p = Path(source)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return SourceType.pdf
    if suffix in (".html", ".htm"):
        return SourceType.html
    return SourceType.text


def _get_ingestor(source_type: SourceType, telegram_limit: int, twitter_max: int = 100):
    from ioc_miner.ingestion import (
        HTMLIngestor, PDFIngestor, PlaintextIngestor, TelegramIngestor, TwitterIngestor,
    )

    match source_type:
        case SourceType.pdf:
            return PDFIngestor()
        case SourceType.html:
            return HTMLIngestor()
        case SourceType.telegram:
            return TelegramIngestor(limit=telegram_limit)
        case SourceType.twitter:
            return TwitterIngestor(max_results=twitter_max)
        case _:
            return PlaintextIngestor()


@app.command()
def extract(
    source: Annotated[str, typer.Argument(help="File path, URL, '-' for stdin, or @channel for Telegram")],
    format: Annotated[OutputFormat, typer.Option("--format", "-f", help="Output format")] = OutputFormat.stdout,
    source_type: Annotated[SourceType, typer.Option("--source", "-s", help="Force source type (default: auto-detect)")] = SourceType.auto,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Write output to file instead of stdout")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="NER model path or HuggingFace ID (enables SecBERT)")] = None,
    no_benign: Annotated[bool, typer.Option("--no-benign", help="Exclude benign/sinkhole IOCs from output")] = False,
    no_unknown: Annotated[bool, typer.Option("--no-unknown", help="Exclude IOCs with unknown verdict")] = False,
    use_ml: Annotated[bool, typer.Option("--use-ml", help="Enable zero-shot NLI classifier for ambiguous IOCs")] = False,
    telegram_limit: Annotated[int, typer.Option("--tg-limit", help="Max Telegram messages to fetch")] = 200,
    twitter_max: Annotated[int, typer.Option("--tw-max", help="Max tweets to fetch")] = 100,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress progress output")] = False,
):
    """Extract and classify IOCs from a file, URL, or stdin."""
    from ioc_miner.extraction import ContextClassifier, NERExtractor, RegexExtractor
    from ioc_miner.models.ioc import IOCVerdict
    from ioc_miner.output import to_csv, to_misp_json, to_stix_json
    from ioc_miner.preprocessing import preprocess, sentence_tokenize

    # ── Ingest ────────────────────────────────────────────────────────────────
    detected = _detect_source_type(source) if source_type == SourceType.auto else source_type
    ingestor = _get_ingestor(detected, telegram_limit, twitter_max)

    if not quiet:
        err_console.print(f"[dim]Ingesting via {detected.value}: {source}[/dim]")

    try:
        raw_text = ingestor.ingest(source)
    except Exception as e:
        err_console.print(f"[red]Ingestion failed:[/red] {e}")
        raise typer.Exit(1)

    # ── Preprocess ────────────────────────────────────────────────────────────
    clean_text = preprocess(raw_text)
    sentences = sentence_tokenize(clean_text)
    if not quiet:
        err_console.print(f"[dim]{len(sentences)} sentences to process[/dim]")

    # ── Extract ───────────────────────────────────────────────────────────────
    regex_ex = RegexExtractor()
    iocs = regex_ex.extract_all(sentences, source=source)

    if model:
        if not quiet:
            err_console.print(f"[dim]Loading NER model: {model}[/dim]")
        ner_ex = NERExtractor(model_path=model)
        ner_iocs = ner_ex.extract_batch(sentences, source=source)
        # Merge: NER results override regex for same (type, value) pairs
        existing = {(i.type, i.value) for i in iocs}
        for ioc in ner_iocs:
            if (ioc.type, ioc.value) not in existing:
                iocs.append(ioc)
                existing.add((ioc.type, ioc.value))

    # ── Classify ──────────────────────────────────────────────────────────────
    if use_ml and not quiet:
        err_console.print("[dim]Loading NLI classifier for ambiguous IOCs...[/dim]")
    classifier = ContextClassifier(use_ml=use_ml)
    iocs = classifier.classify_all(iocs)

    # ── Filter ────────────────────────────────────────────────────────────────
    if no_benign:
        iocs = [i for i in iocs if i.verdict not in (IOCVerdict.BENIGN, IOCVerdict.SINKHOLE)]
    if no_unknown:
        iocs = [i for i in iocs if i.verdict != IOCVerdict.UNKNOWN]

    if not quiet:
        err_console.print(f"[green]{len(iocs)} IOCs extracted[/green]")

    # ── Format ────────────────────────────────────────────────────────────────
    match format:
        case OutputFormat.stix:
            result = to_stix_json(iocs)
        case OutputFormat.csv:
            result = to_csv(iocs)
        case OutputFormat.misp:
            result = to_misp_json(iocs)
        case OutputFormat.stdout:
            result = _render_table(iocs)

    # ── Output ────────────────────────────────────────────────────────────────
    if output:
        output.write_text(result, encoding="utf-8")
        if not quiet:
            err_console.print(f"[green]Saved to {output}[/green]")
    else:
        print(result)


def _render_table(iocs) -> str:
    """Render IOCs as a rich table (stdout mode)."""
    from io import StringIO
    from ioc_miner.models.ioc import IOCVerdict

    _VERDICT_COLOR = {
        IOCVerdict.MALICIOUS: "red",
        IOCVerdict.BENIGN: "green",
        IOCVerdict.SINKHOLE: "yellow",
        IOCVerdict.UNKNOWN: "dim",
    }

    buf = StringIO()
    c = Console(file=buf, highlight=False)
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Type", style="bold", width=10)
    table.add_column("Value", width=45)
    table.add_column("Verdict", width=12)
    table.add_column("Conf", width=6)
    table.add_column("By", width=8)

    for ioc in iocs:
        color = _VERDICT_COLOR.get(ioc.verdict, "white")
        table.add_row(
            ioc.type.value,
            ioc.value,
            f"[{color}]{ioc.verdict.value}[/{color}]",
            f"{ioc.verdict_confidence:.0%}" if ioc.verdict_confidence else "-",
            ioc.extracted_by,
        )

    c.print(table)
    return buf.getvalue()


if __name__ == "__main__":
    app()
