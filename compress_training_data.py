#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


SMART_QUOTES_MAP = {
    "\u2018": "'",  # ‘ left single quote
    "\u2019": "'",  # ’ right single quote
}

DEFAULT_INPUT_DIR = "training_data"


def normalize_quotes(text: str) -> str:
    # Fast, explicit replacement (avoids translating other punctuation).
    for bad, good in SMART_QUOTES_MAP.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def iter_yaml_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.yaml")):
        if path.name == "fact_types.yaml":
            continue
        yield path


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def make_id(rel_path: Path, entry_key: str, idx: int) -> str:
    base = "__".join(rel_path.with_suffix("").parts)
    return f"{base}__{entry_key}__{idx:02d}"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def build_records(input_dir: Path, ingested_at: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in iter_yaml_files(input_dir):
        data = load_yaml(path)
        facts = data.get("facts")
        if not isinstance(facts, dict):
            continue
        entries = facts.get("entries")
        if not isinstance(entries, dict):
            continue

        params = facts.get("params") if isinstance(facts.get("params"), dict) else {}
        rel_path = path.relative_to(input_dir)
        source_mtime = iso_utc(path.stat().st_mtime)
        category = rel_path.parts[0] if rel_path.parts else ""

        for entry_key, variants in entries.items():
            if not isinstance(variants, list):
                continue
            for idx, text in enumerate(variants):
                if not isinstance(text, str):
                    continue
                text = normalize_quotes(text)
                records.append(
                    {
                        "id": make_id(rel_path, entry_key, idx),
                        "type": entry_key,
                        "text": text,
                        "params": params,
                        "category": category,
                        "source_file": str(rel_path),
                        "source_mtime": source_mtime,
                        "ingested_at": ingested_at,
                    }
                )
    return records


def build_fact_lines(input_dir: Path) -> List[str]:
    lines: List[str] = []
    for path in iter_yaml_files(input_dir):
        data = load_yaml(path)
        facts = data.get("facts")
        if not isinstance(facts, dict):
            continue
        entries = facts.get("entries")
        if not isinstance(entries, dict):
            continue

        for variants in entries.values():
            if not isinstance(variants, list):
                continue
            for text in variants:
                if not isinstance(text, str):
                    continue
                lines.append(normalize_quotes(text))
    return lines


def write_jsonl(output_path: Path, records: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_txt(output_path: Path, lines: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert training_data YAML facts into a compressed text format."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        required=False,
        help="Root directory containing YAML fact files (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        choices=("txt", "jsonl"),
        required=True,
        help="Output format.",
    )
    parser.add_argument(
        "--output",
        help="Output path. Defaults to training_data.<format>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path(f"training_data.{args.format}").resolve()
    )

    if args.format == "txt":
        lines = build_fact_lines(input_dir)
        write_txt(output_path, lines)
        print(f"Wrote {len(lines)} facts to {output_path}")
        return 0

    ingested_at = datetime.now(timezone.utc).isoformat()
    records = build_records(input_dir, ingested_at)
    write_jsonl(output_path, records)
    print(f"Wrote {len(records)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
