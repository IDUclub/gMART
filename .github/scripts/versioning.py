"""Version policy helpers shared by the PR autofill, dev release and main release workflows.

Every merge into dev bumps the version: a PR labelled ``major`` raises the major part, a
``feat/`` or ``feature/`` branch the minor part, any other branch the patch part. The new
version is written to ``pyproject.toml`` and the ``version_files`` of ``[tool.commitizen]``,
and ``CHANGELOG.md`` gets a section for it. A release to main does not bump: it tags the
version that came from dev and collects the changelog sections since the previous tag.

Standard library only, so the workflows need no dependencies to run it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path("pyproject.toml")
CHANGELOG = Path("CHANGELOG.md")
VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
SECTION = re.compile(r"^## v(\d+\.\d+\.\d+)\b", re.M)
MINOR_BRANCHES = ("feat/", "feature/")


def parse(version: str) -> tuple[int, int, int]:
    match = VERSION.match(version.strip())
    if match is None:
        raise SystemExit(f"not a X.Y.Z version: {version!r}")
    return tuple(int(part) for part in match.groups())


def text(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def project_version() -> str:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]


def tags() -> list[tuple[int, int, int]]:
    out = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True
    ).stdout
    return sorted(parse(tag) for tag in out.split() if VERSION.match(tag))


def kind(branch: str, labels: str) -> str:
    if "major" in {label.strip().lower() for label in labels.split(",")}:
        return "major"
    return "minor" if branch.lower().startswith(MINOR_BRANCHES) else "patch"


def bump(version: tuple[int, int, int], increment: str) -> tuple[int, int, int]:
    major, minor, patch = version
    if increment == "major":
        return major + 1, 0, 0
    if increment == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def next_version(increment: str) -> str:
    # A release tag can be ahead of the project version (bumps once made on main), and
    # a new version must never repeat or undercut a released one.
    known = [parse(project_version()), *tags()]
    return text(bump(max(known), increment))


def version_files() -> list[tuple[Path, re.Pattern | None]]:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    entries = config.get("tool", {}).get("commitizen", {}).get("version_files", [])
    files = []
    for entry in entries:
        path, _, pattern = entry.partition(":")
        files.append((Path(path), re.compile(pattern) if pattern else None))
    return files


def apply(new: str) -> None:
    """Write ``new`` everywhere the current version is declared."""
    old = project_version()
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    # [project].version and, where commitizen keeps its own copy, [tool.commitizen].version.
    updated = re.sub(
        rf'^(version\s*=\s*"){re.escape(old)}(")', rf"\g<1>{new}\g<2>", pyproject, flags=re.M
    )
    if updated == pyproject:
        raise SystemExit(f"version {old} not found in {PYPROJECT}")
    PYPROJECT.write_text(updated, encoding="utf-8")
    for path, pattern in version_files():
        if path == PYPROJECT:
            continue
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        changed = False
        for index, line in enumerate(lines):
            if (pattern is None or pattern.search(line)) and old in line:
                lines[index] = line.replace(old, new)
                changed = True
        if not changed:
            raise SystemExit(f"version {old} not found in {path}")
        path.write_text("".join(lines), encoding="utf-8")


def changelog(new: str, date: str, title: str, items: list[str]) -> None:
    """Put the section of ``new`` on top of CHANGELOG.md."""
    entries = "\n".join(f"- {item}" for item in items) or "- (no commits)"
    section = f"## v{new} ({date})\n\n{title}\n\n{entries}\n\n"
    current = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.exists() else ""
    first = SECTION.search(current)
    head, rest = (current[: first.start()], current[first.start() :]) if first else (current, "")
    CHANGELOG.write_text(head + section + rest, encoding="utf-8")


def notes(version: str, since: str | None) -> str:
    """Changelog sections after ``since`` up to and including ``version``."""
    if not CHANGELOG.exists():
        return ""
    current = CHANGELOG.read_text(encoding="utf-8")
    top, low = parse(version), parse(since) if since else None
    starts = list(SECTION.finditer(current))
    picked = []
    for index, match in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(current)
        number = parse(match.group(1))
        if number <= top and (low is None or number > low):
            picked.append(current[match.start() : end].strip())
    return "\n\n".join(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("current", help="project version")
    latest = commands.add_parser("latest-tag", help="newest v* tag version")
    latest.add_argument("--below", help="only tags older than this version")
    increment = commands.add_parser("kind", help="major | minor | patch for a PR")
    increment.add_argument("--branch", required=True)
    increment.add_argument("--labels", default="")
    upcoming = commands.add_parser("next", help="version after this increment")
    upcoming.add_argument("--kind", choices=["major", "minor", "patch"], required=True)
    write = commands.add_parser("apply", help="write a version to the version files")
    write.add_argument("--version", required=True)
    section = commands.add_parser("changelog", help="add a CHANGELOG.md section")
    section.add_argument("--version", required=True)
    section.add_argument("--date", required=True)
    section.add_argument("--title", required=True)
    section.add_argument("--items-file", type=Path, required=True)
    release = commands.add_parser("notes", help="changelog sections of a release")
    release.add_argument("--version", required=True)
    release.add_argument("--since")
    args = parser.parse_args()

    if args.command == "current":
        print(project_version())
    elif args.command == "latest-tag":
        below = parse(args.below) if args.below else None
        older = [tag for tag in tags() if below is None or tag < below]
        if older:
            print(text(older[-1]))
    elif args.command == "kind":
        print(kind(args.branch, args.labels))
    elif args.command == "next":
        print(next_version(args.kind))
    elif args.command == "apply":
        parse(args.version)
        apply(args.version)
    elif args.command == "changelog":
        items = [
            line.strip()
            for line in args.items_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        changelog(args.version, args.date, args.title, items)
    elif args.command == "notes":
        sys.stdout.write(notes(args.version, args.since))


if __name__ == "__main__":
    main()
