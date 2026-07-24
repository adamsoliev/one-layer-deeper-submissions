#!/usr/bin/env python3
"""Submit the committed model, persist its result, and refresh the README."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys

import duckdb


ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = ROOT / "submissions.duckdb"
README_PATH = ROOT / "README.md"
SCHEMA_PATH = ROOT / "schema.sql"
SUBMISSION_PATH = ROOT / "submission.py"
REPOSITORY_URL = "https://github.com/adamsoliev/one-layer-deeper-submissions"
TABLE_START = "<!-- SUBMISSIONS_TABLE_START -->"
TABLE_END = "<!-- SUBMISSIONS_TABLE_END -->"


@dataclass
class ParsedResult:
    valid: bool = False
    validated_bytes: int | None = None
    queued: bool = False
    tier: str = ""
    dataset_id: str | None = None
    dataset_label: str | None = None
    attempts_left: int | None = None
    submission_id: str | None = None
    view_url: str | None = None
    status: str = "failed"
    score_pct: float | None = None
    max_t: str | None = None
    ood_n_max_t: str | None = None
    suite: str | None = None
    run_id: str | None = None
    modal_call_id: str | None = None


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_committed_worktree() -> str:
    try:
        commit_hash = run_git("rev-parse", "HEAD")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("create at least one Git commit before submitting") from exc
    if run_git("status", "--porcelain"):
        raise RuntimeError(
            "the Git worktree must be clean so HEAD identifies the submitted source"
        )
    tracked = run_git("ls-files", "--error-unmatch", SUBMISSION_PATH.name)
    if tracked != SUBMISSION_PATH.name:
        raise RuntimeError("submission.py must be tracked by Git")
    return commit_hash


def parse_output(output: str, requested_tier: str, dataset_id: str | None) -> ParsedResult:
    result = ParsedResult(tier=requested_tier, dataset_id=dataset_id)
    for line in output.splitlines():
        valid_match = re.match(r"^valid:\s+.+\s+\((\d+) bytes\)$", line)
        if valid_match:
            result.valid = True
            result.validated_bytes = int(valid_match.group(1))
            continue

        if line.startswith("queued:"):
            result.queued = True
            result.submission_id = line.partition(":")[2].strip()
            continue
        if line.startswith("tier:"):
            result.tier = line.partition(":")[2].strip()
            continue
        if line.startswith("data:"):
            result.dataset_id = line.partition(":")[2].strip()
            continue
        if line.startswith("left:"):
            result.attempts_left = int(line.partition(":")[2].split()[0])
            continue
        if line.startswith("view:"):
            result.view_url = line.partition(":")[2].strip()
            continue

        status_match = re.match(r"^\[(queued|running|succeeded|failed)\]", line)
        if status_match:
            result.status = status_match.group(1)
            continue

        depth_match = re.match(r"^(OOD N max T|max T)\s+(.+)$", line)
        if depth_match:
            if depth_match.group(1) == "OOD N max T":
                result.ood_n_max_t = depth_match.group(2)
            else:
                result.max_t = depth_match.group(2)
            continue

        field_match = re.match(r"^([A-Za-z][A-Za-z ]*?)\s{2,}(.+)$", line)
        if not field_match:
            continue
        field = field_match.group(1).strip().lower()
        value = field_match.group(2).strip()
        if field == "submission":
            result.submission_id = value
        elif field == "status":
            result.status = value
        elif field == "score" and value != "—":
            result.score_pct = float(value.removesuffix("%"))
        elif field == "tier":
            result.tier = value
        elif field == "dataset":
            result.dataset_label = value
        elif field == "suite":
            result.suite = value
        elif field == "run":
            result.run_id = value
        elif field == "modal call":
            result.modal_call_id = value
    return result


def submit(args: argparse.Namespace) -> tuple[str, int, str]:
    command = [
        "one-layer",
        "submit",
        SUBMISSION_PATH.name,
        "--tier",
        args.tier,
        "--wait",
    ]
    if args.dataset:
        command.extend(["--dataset", args.dataset])
    if args.server:
        command.extend(["--server", args.server])

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return "".join(lines), process.wait(), shlex.join(command)


def read_existing_output(args: argparse.Namespace) -> tuple[str, int, str]:
    output = args.from_output.read_text(encoding="utf-8")
    print(output, end="" if output.endswith("\n") else "\n")
    return output, 0, "historical import"


def persist(
    *,
    note: str,
    commit_hash: str,
    requested_tier: str,
    requested_dataset: str | None,
    output: str,
    exit_code: int,
    command: str,
) -> int:
    parsed = parse_output(output, requested_tier, requested_dataset)
    file_sha256 = hashlib.sha256(SUBMISSION_PATH.read_bytes()).hexdigest()
    with duckdb.connect(str(DATABASE_PATH)) as connection:
        connection.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        number = connection.execute(
            "SELECT coalesce(max(number), 0) + 1 FROM submissions"
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO submissions VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            [
                number,
                note,
                commit_hash,
                SUBMISSION_PATH.name,
                file_sha256,
                datetime.now(timezone.utc),
                parsed.valid,
                parsed.validated_bytes,
                parsed.queued,
                parsed.tier,
                parsed.dataset_id,
                parsed.dataset_label,
                parsed.attempts_left,
                parsed.submission_id,
                parsed.view_url,
                parsed.status,
                parsed.score_pct,
                parsed.max_t,
                parsed.ood_n_max_t,
                parsed.suite,
                parsed.run_id,
                parsed.modal_call_id,
                exit_code,
                command,
                output,
            ],
        )
    return number


def markdown_escape(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def refresh_readme() -> None:
    with duckdb.connect(str(DATABASE_PATH), read_only=True) as connection:
        rows = connection.execute(
            """
            SELECT number, note, commit_hash, status, score_pct
            FROM submissions
            ORDER BY number
            """
        ).fetchall()

    table = [
        "| Number | Note | Commit | Status | Score |",
        "| ---: | --- | --- | --- | ---: |",
    ]
    for number, note, commit_hash, status, score_pct in rows:
        commit = (
            f"[`{commit_hash[:7]}`]({REPOSITORY_URL}/commit/{commit_hash})"
        )
        score = "—" if score_pct is None else f"{score_pct:.2f}%"
        table.append(
            f"| {number} | {markdown_escape(note)} | {commit} | "
            f"{status} | {score} |"
        )

    readme = README_PATH.read_text(encoding="utf-8")
    before, separator, remainder = readme.partition(TABLE_START)
    if not separator:
        raise RuntimeError(f"{TABLE_START} is missing from README.md")
    _, separator, after = remainder.partition(TABLE_END)
    if not separator:
        raise RuntimeError(f"{TABLE_END} is missing from README.md")
    rendered = f"{before}{TABLE_START}\n" + "\n".join(table)
    rendered += f"\n{TABLE_END}{after}"
    README_PATH.write_text(rendered, encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit the committed model and record its complete result."
    )
    parser.add_argument("--note", required=True, help="experiment description")
    parser.add_argument(
        "--tier",
        required=True,
        choices=("easy", "medium", "hard"),
    )
    parser.add_argument("--dataset", help="required for Easy and Medium")
    parser.add_argument("--server", help="override the competition service URL")
    parser.add_argument(
        "--from-output",
        type=Path,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    if args.tier in {"easy", "medium"} and not args.dataset:
        raise SystemExit("--dataset is required for Easy and Medium")
    commit_hash = require_committed_worktree()
    if args.from_output:
        output, exit_code, command = read_existing_output(args)
    else:
        output, exit_code, command = submit(args)
    number = persist(
        note=args.note,
        commit_hash=commit_hash,
        requested_tier=args.tier,
        requested_dataset=args.dataset,
        output=output,
        exit_code=exit_code,
        command=command,
    )
    refresh_readme()
    print(f"recorded submission {number} in {DATABASE_PATH.name}")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
