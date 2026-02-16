#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STAGES_DIR = Path(__file__).resolve().parent / "stages"


def discover_stages() -> Dict[str, Path]:
    stages: Dict[str, Path] = {}
    for path in sorted(STAGES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        stages[path.stem] = path
    return stages


def resolve_stage(stage_name: str, stages: Dict[str, Path]) -> Optional[Path]:
    normalized = stage_name[:-3] if stage_name.endswith(".py") else stage_name
    if normalized in stages:
        return stages[normalized]
    if normalized.isdigit():
        matches = [
            path
            for stem, path in stages.items()
            if stem.split("_", 1)[0] == normalized
        ]
        if len(matches) == 1:
            return matches[0]
    for path in stages.values():
        if path.name == stage_name:
            return path
    return None


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Run a specific pipeline stage from the stages/ directory."
    )
    parser.add_argument(
        "stage",
        help="Stage to run (for example: 00, 00_preprocess, or 00_preprocess.py).",
    )
    return parser.parse_known_args()


def main() -> int:
    args, stage_args = parse_args()
    stages = discover_stages()
    stage_path = resolve_stage(args.stage, stages)
    if stage_path is None:
        print(f"Unknown stage: {args.stage}", file=sys.stderr)
        available = ", ".join(stages.keys()) if stages else "(none found)"
        print(f"Available stages: {available}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent
    cmd = [sys.executable, str(stage_path), *stage_args]
    completed = subprocess.run(cmd, cwd=repo_root)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
