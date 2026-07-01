"""Prepare JSONL data for the probability-distribution poem generator.

Input files:
- data/classic_poems_raw.jsonl
- data/experience_poem_pairs.jsonl

Output file:
- data/train.jsonl

Project rules:
- Input is a user experience prompt.
- Target output is always a 3-line Korean poem.
- No hardcoded poem templates or demo fallback data are generated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REQUIRED_PAIR_FIELDS = ("experience", "poem")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def normalize_poem(poem: str) -> str:
    lines = [line.strip() for line in poem.splitlines() if line.strip()]
    return "\n".join(lines)


def validate_pair(row: dict[str, Any], source_path: Path, index: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_PAIR_FIELDS if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"Missing {missing} in {source_path} row {index}")

    poem = normalize_poem(str(row["poem"]))
    lines = [line for line in poem.splitlines() if line.strip()]
    if len(lines) != 3:
        raise ValueError(f"Poem must have exactly 3 non-empty lines in {source_path} row {index}; got {len(lines)}")

    return {
        "experience": str(row["experience"]).strip(),
        "poem": poem,
        "source_type": str(row.get("source_type", "unknown")).strip() or "unknown",
        "source_title": str(row.get("source_title", "")).strip(),
        "style_tags": row.get("style_tags", []),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_train_rows(classic_rows: list[dict[str, Any]], experience_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, source_rows in (("classic_poems_raw.jsonl", classic_rows), ("experience_poem_pairs.jsonl", experience_rows)):
        source_path = Path("data") / source_name
        for index, row in enumerate(source_rows, start=1):
            rows.append(validate_pair(row, source_path, index))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classic", type=Path, default=Path("data/classic_poems_raw.jsonl"))
    parser.add_argument("--experience", type=Path, default=Path("data/experience_poem_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"))
    args = parser.parse_args()

    classic_rows = read_jsonl(args.classic)
    experience_rows = read_jsonl(args.experience)
    train_rows = build_train_rows(classic_rows, experience_rows)
    write_jsonl(args.output, train_rows)

    print(f"classic_rows={len(classic_rows)}")
    print(f"experience_rows={len(experience_rows)}")
    print(f"train_rows={len(train_rows)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
