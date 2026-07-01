"""Validate the master poem dataset and build train.jsonl.

Input:
- data/experience_poem_pairs.jsonl

Outputs:
- data/train.jsonl
- data/dataset_issues.jsonl

Project rules:
- Input is a user experience prompt.
- Output is always a 3-line Korean poem.
- Validation must not invent or replace poems.
- Only rows that pass validation are converted to the training messages format.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = "너는 경험을 3줄 한국어 시로 바꾸는 시 생성 모델이다."

REQUIRED_FIELDS = (
    "id",
    "source_type",
    "source_title",
    "source_author",
    "source_text",
    "experience",
    "poem",
    "style_tags",
)

ALLOWED_SOURCE_TYPES = {"classic_poem", "modern_daily"}
PROSE_LIKE_PATTERNS = (
    "입니다",
    "합니다",
    "하였다",
    "했다",
    "것이다",
    "수 있다",
    "보여준다",
    "의미한다",
    "상징한다",
    "나타낸다",
    "설명한다",
)
PROMPT_MARKERS = ("경험:", "시:", "해설", "제목:", "조건:", "```")


@dataclass
class Issue:
    row_number: int
    row_id: str
    severity: str
    reason: str
    detail: str = ""
    row: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_number": self.row_number,
            "id": self.row_id,
            "severity": self.severity,
            "reason": self.reason,
            "detail": self.detail,
            "row": self.row,
        }


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[Issue]]:
    rows: list[dict[str, Any]] = []
    issues: list[Issue] = []

    if not path.exists():
        issues.append(Issue(0, "", "error", "input_file_not_found", str(path)))
        return rows, issues

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                issues.append(Issue(line_number, "", "error", "invalid_json", str(exc), {"raw": raw[:500]}))
                continue
            if not isinstance(row, dict):
                issues.append(Issue(line_number, "", "error", "row_is_not_object", "Each JSONL line must be an object."))
                continue
            rows.append(row | {"__line_number": line_number})
    return rows, issues


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def poem_lines(poem: str) -> list[str]:
    return [line.strip() for line in poem.splitlines() if line.strip()]


def is_style_tags_valid(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item.strip() for item in value)


def has_prose_like_ending(poem: str) -> bool:
    prose_hits = sum(poem.count(pattern) for pattern in PROSE_LIKE_PATTERNS)
    return prose_hits >= 2


def has_prompt_marker_leak(poem: str) -> bool:
    return any(marker in poem for marker in PROMPT_MARKERS)


def validate_row(row: dict[str, Any], seen_experiences: set[str]) -> tuple[bool, list[Issue]]:
    row_number = int(row.get("__line_number", 0))
    row_id = str(row.get("id", "")).strip()
    issues: list[Issue] = []

    def add(reason: str, detail: str = "", severity: str = "error") -> None:
        clean_row = {k: v for k, v in row.items() if k != "__line_number"}
        issues.append(Issue(row_number, row_id, severity, reason, detail, clean_row))

    for field_name in REQUIRED_FIELDS:
        if field_name not in row:
            add("missing_field", field_name)

    if not row_id:
        add("empty_id")

    source_type = str(row.get("source_type", "")).strip()
    if source_type not in ALLOWED_SOURCE_TYPES:
        add("invalid_source_type", source_type)

    experience = str(row.get("experience", "")).strip()
    if not experience:
        add("empty_experience")
    else:
        normalized_experience = normalize_space(experience)
        if normalized_experience in seen_experiences:
            add("duplicate_experience", normalized_experience)
        else:
            seen_experiences.add(normalized_experience)
        if len(experience) < 8:
            add("experience_too_short", experience, severity="warning")
        if len(experience) > 180:
            add("experience_too_long", f"{len(experience)} chars", severity="warning")

    poem = str(row.get("poem", "")).strip()
    if not poem:
        add("empty_poem")
    else:
        lines = poem_lines(poem)
        if len(lines) != 3:
            add("not_three_lines", f"got {len(lines)} lines")
        for index, line in enumerate(lines, start=1):
            if len(line) > 45:
                add("poem_line_too_long", f"line {index}: {len(line)} chars", severity="warning")
        if len(poem.replace("\n", "")) > 150:
            add("poem_too_long", f"{len(poem)} chars", severity="warning")
        if has_prose_like_ending(poem):
            add("prose_report_like_output", "prose-like endings appeared repeatedly")
        if has_prompt_marker_leak(poem):
            add("prompt_marker_leak")

    style_tags = row.get("style_tags", [])
    if not is_style_tags_valid(style_tags):
        add("invalid_style_tags", "style_tags must be a list of non-empty strings")

    # Warnings do not block conversion. Errors do.
    has_error = any(issue.severity == "error" for issue in issues)
    return not has_error, issues


def to_training_message(row: dict[str, Any]) -> dict[str, Any]:
    experience = str(row["experience"]).strip()
    poem = "\n".join(poem_lines(str(row["poem"])))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"경험: {experience}"},
            {"role": "assistant", "content": poem},
        ]
    }


def summarize(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], issues: list[Issue]) -> None:
    source_counter = Counter(str(row.get("source_type", "unknown")) for row in rows)
    issue_counter = Counter(issue.reason for issue in issues)
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)

    print("=== dataset summary ===")
    print(f"master_rows={len(rows)}")
    print(f"train_rows={len(train_rows)}")
    print(f"source_type_counts={dict(source_counter)}")
    print(f"errors={error_count}")
    print(f"warnings={warning_count}")
    print(f"issue_counts={dict(issue_counter)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate poem dataset and build training JSONL.")
    parser.add_argument("--input", type=Path, default=Path("data/experience_poem_pairs.jsonl"))
    parser.add_argument("--train-output", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--issues-output", type=Path, default=Path("data/dataset_issues.jsonl"))
    parser.add_argument("--fail-on-warning", action="store_true", help="Treat warning rows as blocked from train.jsonl.")
    args = parser.parse_args()

    rows, read_issues = read_jsonl(args.input)
    all_issues = list(read_issues)
    seen_experiences: set[str] = set()
    train_rows: list[dict[str, Any]] = []

    for row in rows:
        valid, row_issues = validate_row(row, seen_experiences)
        all_issues.extend(row_issues)
        if args.fail_on_warning and row_issues:
            valid = False
        if valid:
            train_rows.append(to_training_message(row))

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.issues_output, [issue.to_dict() for issue in all_issues])
    summarize(rows, train_rows, all_issues)
    print(f"train_output={args.train_output}")
    print(f"issues_output={args.issues_output}")


if __name__ == "__main__":
    main()
