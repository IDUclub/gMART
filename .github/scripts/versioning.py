"""Version policy helpers shared by the PR autofill, version bump and release workflows.

Every PR into dev is merged with a new version, bumped in its branch when auto-merge is
enabled: a PR labelled ``major`` raises the major part, a ``feat/`` or ``feature/`` branch the
minor part, any other branch the patch part. The version is written to ``pyproject.toml`` and
the ``version_files`` of ``[tool.commitizen]``, and ``CHANGELOG.md`` gets a section for the PR.
A release to main does not bump: it tags the version that came from dev and collects the
changelog sections since the previous tag.

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


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def read(path: Path) -> str:
    # Bytes in, bytes out: line endings stay as the repository has them.
    return path.read_bytes().decode("utf-8")


def write(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))


def project_version(ref: str | None = None) -> str:
    """The version in pyproject.toml of the work tree, or of a git ref such as origin/dev."""
    source = git("show", f"{ref}:{PYPROJECT.as_posix()}") if ref else read(PYPROJECT)
    return tomllib.loads(source)["project"]["version"]


def tags() -> list[tuple[int, int, int]]:
    return sorted(
        parse(tag) for tag in git("tag", "--list", "v*").split() if VERSION.match(tag)
    )


def released(ref: str | None) -> tuple[int, int, int]:
    # A release tag can be ahead of the project version (bumps once made on main), and a new
    # version must never repeat or undercut a released one.
    return max([parse(project_version(ref)), *tags()])


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


def version_files() -> list[tuple[Path, re.Pattern | None]]:
    config = tomllib.loads(read(PYPROJECT))
    entries = config.get("tool", {}).get("commitizen", {}).get("version_files", [])
    files = []
    for entry in entries:
        path, _, pattern = entry.partition(":")
        files.append((Path(path), re.compile(pattern) if pattern else None))
    return files


def apply(new: str) -> None:
    """Write ``new`` everywhere the current version is declared."""
    old = project_version()
    if old == new:
        return
    pyproject = read(PYPROJECT)
    # [project].version and, where commitizen keeps its own copy, [tool.commitizen].version.
    updated = re.sub(
        rf'^(version\s*=\s*"){re.escape(old)}(")',
        rf"\g<1>{new}\g<2>",
        pyproject,
        flags=re.M,
    )
    if updated == pyproject:
        raise SystemExit(f"version {old} not found in {PYPROJECT}")
    write(PYPROJECT, updated)
    for path, pattern in version_files():
        if path == PYPROJECT:
            continue
        lines = read(path).splitlines(keepends=True)
        changed = False
        for index, line in enumerate(lines):
            if (pattern is None or pattern.search(line)) and old in line:
                lines[index] = line.replace(old, new)
                changed = True
        if not changed:
            raise SystemExit(f"version {old} not found in {path}")
        write(path, "".join(lines))


def sections(current: str) -> list[tuple[str, int, int]]:
    """(version, start, end) of every ``## vX.Y.Z`` section."""
    starts = list(SECTION.finditer(current))
    return [
        (
            match.group(1),
            match.start(),
            starts[index + 1].start() if index + 1 < len(starts) else len(current),
        )
        for index, match in enumerate(starts)
    ]


def changelog(new: str, date: str, title: str, items: list[str]) -> None:
    """Put the section of ``new`` on top of CHANGELOG.md.

    A section this PR wrote before (same ``title``, e.g. a bump redone after dev moved) is
    replaced instead of repeated.
    """
    current = read(CHANGELOG) if CHANGELOG.exists() else ""
    for _, start, end in reversed(sections(current)):
        body = current[start:end].splitlines()  # splitlines() also drops a CR
        if len(body) > 2 and body[2] == title:
            current = current[:start] + current[end:]
    entries = "\n".join(f"- {item}" for item in items) or "- (no commits)"
    section = f"## v{new} ({date})\n\n{title}\n\n{entries}\n\n"
    first = SECTION.search(current)
    head, rest = (
        (current[: first.start()], current[first.start() :]) if first else (current, "")
    )
    write(CHANGELOG, head + section + rest)


def notes(version: str, since: str | None) -> str:
    """Changelog sections after ``since`` up to and including ``version``."""
    if not CHANGELOG.exists():
        return ""
    current = read(CHANGELOG)
    top, low = parse(version), parse(since) if since else None
    picked = [
        current[start:end].strip()
        for number, start, end in sections(current)
        if parse(number) <= top and (low is None or parse(number) > low)
    ]
    return "\n\n".join(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    current = commands.add_parser("current", help="project version")
    current.add_argument(
        "--ref", help="read it from this git ref instead of the work tree"
    )
    latest = commands.add_parser("latest-tag", help="newest v* tag version")
    latest.add_argument("--below", help="only tags older than this version")
    increment = commands.add_parser("kind", help="major | minor | patch for a PR")
    increment.add_argument("--branch", required=True)
    increment.add_argument("--labels", default="")
    upcoming = commands.add_parser("next", help="version after this increment")
    upcoming.add_argument("--kind", choices=["major", "minor", "patch"], required=True)
    upcoming.add_argument(
        "--ref", help="bump the version of this git ref (e.g. origin/dev)"
    )
    ahead = commands.add_parser(
        "bumped",
        help="exit 0 when the work tree version is past the base and every release",
    )
    ahead.add_argument("--ref", required=True)
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
        print(project_version(args.ref))
    elif args.command == "latest-tag":
        below = parse(args.below) if args.below else None
        older = [tag for tag in tags() if below is None or tag < below]
        if older:
            print(text(older[-1]))
    elif args.command == "kind":
        print(kind(args.branch, args.labels))
    elif args.command == "next":
        print(text(bump(released(args.ref), args.kind)))
    elif args.command == "bumped":
        sys.exit(0 if parse(project_version()) > released(args.ref) else 1)
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
