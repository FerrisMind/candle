#!/usr/bin/env python3
"""
Audit source files by size for refactoring pressure.

Zones (physical line count, including blanks/comments):
  small     < 150     — fine; not a size smell
  green     150–500   — ideal module size
  yellow    500–1000  — large but OK for one cohesive entity
  red       1000–2000 — split signal: mixed responsibilities likely
  critical  > 2000    — too large (tests/macros/codegen exceptions apply)

Extra heuristics (printed as hints, not hard fails):
  * Section banner comments: // ----- Foo -----  or  // ==== Foo ====
  * Large inline #[cfg(test)] modules that could move to tests/

Usage:
  python scripts/refactor_size_audit.py
  python scripts/refactor_size_audit.py candle-core/src --ext .rs
  python scripts/refactor_size_audit.py --min-zone yellow --json
  python scripts/refactor_size_audit.py --top 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

# Inclusive lower bounds for each named zone (upper is next lower - 1, or open).
ZONE_ORDER = ("small", "green", "yellow", "red", "critical")
ZONE_BOUNDS = {
    "small": (0, 149),
    "green": (150, 500),
    "yellow": (501, 1000),
    "red": (1001, 2000),
    "critical": (2001, None),
}
ZONE_LABEL = {
    "small": "small (<150)",
    "green": "green (150-500) - ideal",
    "yellow": "yellow (500-1000) - large but OK",
    "red": "red (1000-2000) - split signal",
    "critical": "critical (>2000) - too large",
}
# ASCII markers — Windows consoles often cannot encode emoji (cp1251).
ZONE_MARK = {
    "small": "[ ]",
    "green": "[G]",
    "yellow": "[Y]",
    "red": "[R]",
    "critical": "[!]",
}

DEFAULT_EXCLUDES = (
    ".git",
    "target",
    "node_modules",
    ".swarm",
    "candle_refs",
    "__pycache__",
    ".venv",
    "venv",
)

# // ----- Validation -----  |  // ==== Network ====  |  // --- helpers ---
SECTION_BANNER_RE = re.compile(
    r"^\s*//\s*(?:[-*=]{3,}|#{3,})\s*.+\s*(?:[-*=]{3,}|#{3,})\s*$"
)
CFG_TEST_RE = re.compile(r"#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]")


@dataclass(frozen=True)
class FileReport:
    path: str
    lines: int
    non_blank: int
    zone: str
    section_banners: int
    inline_cfg_test: bool
    hints: Tuple[str, ...]


def classify(lines: int) -> str:
    if lines < 150:
        return "small"
    if lines <= 500:
        return "green"
    if lines <= 1000:
        return "yellow"
    if lines <= 2000:
        return "red"
    return "critical"


def zone_rank(zone: str) -> int:
    return ZONE_ORDER.index(zone)


def should_skip(path: Path, excludes: Sequence[str]) -> bool:
    parts = set(path.parts)
    return any(ex in parts for ex in excludes)


def iter_files(
    roots: Sequence[Path],
    extensions: Sequence[str],
    excludes: Sequence[str],
) -> Iterator[Path]:
    exts = {e if e.startswith(".") else f".{e}" for e in extensions}
    for root in roots:
        root = root.resolve()
        if root.is_file():
            if root.suffix in exts and not should_skip(root, excludes):
                yield root
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in exts:
                continue
            if should_skip(path, excludes):
                continue
            yield path


def analyze_file(path: Path, repo: Path) -> FileReport:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        rel = str(path)
        return FileReport(
            path=rel,
            lines=0,
            non_blank=0,
            zone="small",
            section_banners=0,
            inline_cfg_test=False,
            hints=(f"unreadable: {exc}",),
        )

    lines_list = text.splitlines()
    n_lines = len(lines_list)
    n_non_blank = sum(1 for line in lines_list if line.strip())
    banners = sum(1 for line in lines_list if SECTION_BANNER_RE.match(line))
    has_cfg_test = bool(CFG_TEST_RE.search(text))

    hints: List[str] = []
    zone = classify(n_lines)
    if banners >= 2:
        hints.append(
            f"{banners} section-banner comments - likely multiple responsibilities; split by banner"
        )
    elif banners == 1 and zone_rank(zone) >= zone_rank("yellow"):
        hints.append("section-banner comment in a large file - consider extracting that section")

    if has_cfg_test and n_lines > 1000:
        hints.append(
            "inline #[cfg(test)] in a large file - move tests to tests/ or a sibling *_tests.rs"
        )

    if zone == "critical":
        hints.append(
            "critical size: check for inlined tests, missing macros, or generated code before refactoring"
        )
    elif zone == "red":
        hints.append(
            "red zone: prefer one concern per module; use F2 / file outline if you scroll >2-3 screens"
        )

    try:
        rel = str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        rel = str(path)

    return FileReport(
        path=rel.replace("\\", "/"),
        lines=n_lines,
        non_blank=n_non_blank,
        zone=zone,
        section_banners=banners,
        inline_cfg_test=has_cfg_test,
        hints=tuple(hints),
    )


def filter_by_min_zone(reports: Iterable[FileReport], min_zone: str) -> List[FileReport]:
    floor = zone_rank(min_zone)
    return [r for r in reports if zone_rank(r.zone) >= floor]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score files for refactoring pressure by line count."
    )
    p.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="files or directories to scan (default: repo root)",
    )
    p.add_argument(
        "--ext",
        action="append",
        default=None,
        help="extension to include (repeatable). Default: .rs",
    )
    p.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="directory name to skip (repeatable). Defaults include target, .git, …",
    )
    p.add_argument(
        "--min-zone",
        choices=ZONE_ORDER,
        default="yellow",
        help="only list files at this zone or worse (default: yellow). Summary always full.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="list every scanned file (same as --min-zone small)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        help="after filtering, keep only the N largest files",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="emit JSON (summary + files)",
    )
    p.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="repo root for relative paths (default: cwd)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo = (args.repo or Path.cwd()).resolve()
    roots = [(repo / p).resolve() if not Path(p).is_absolute() else Path(p).resolve() for p in args.paths]
    extensions = args.ext if args.ext else [".rs"]
    excludes = list(DEFAULT_EXCLUDES)
    if args.exclude:
        excludes.extend(args.exclude)

    all_reports = [
        analyze_file(path, repo) for path in sorted(iter_files(roots, extensions, excludes))
    ]
    min_zone = "small" if args.all else args.min_zone
    listed = filter_by_min_zone(all_reports, min_zone)
    listed.sort(key=lambda r: (-r.lines, r.path))
    if args.top > 0:
        listed = listed[: args.top]

    summary = {zone: sum(1 for r in all_reports if r.zone == zone) for zone in ZONE_ORDER}

    if args.json:
        payload = {
            "summary": summary,
            "bounds": {
                k: {"min": v[0], "max": v[1]} for k, v in ZONE_BOUNDS.items()
            },
            "files": [asdict(r) for r in listed],
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
        return 0

    # Human summary uses full scan; detail uses filtered list.
    print("Refactor size audit")
    print("===================")
    print(f"files scanned: {len(all_reports)}")
    for zone in ZONE_ORDER:
        print(f"  {ZONE_MARK[zone]} {ZONE_LABEL[zone]}: {summary[zone]}")
    print()
    print(f"listing --min-zone {min_zone}" + (f" --top {args.top}" if args.top else ""))
    print(f"{'zone':10} {'lines':>6} {'nb':>6}  path")
    print("-" * 72)
    if not listed:
        print("(no files in selected zones)")
        return 0
    for r in listed:
        print(f"{ZONE_MARK[r.zone]} {r.zone:8} {r.lines:6d} {r.non_blank:6d}  {r.path}")
        for hint in r.hints:
            print(f"           hint: {hint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
