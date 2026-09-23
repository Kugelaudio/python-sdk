"""Agent skills bundled with the KugelAudio SDK.

Coding agents (Claude Code, Cursor, Codex) load skills from a skills directory
such as ``.claude/skills/``; they do not look inside installed packages. This
module ships the skill files and a ``kugelaudio-skills`` command that copies
them where an agent will find them::

    kugelaudio-skills install              # ./.claude/skills/<name>/
    kugelaudio-skills install --global     # ~/.claude/skills/<name>/
    kugelaudio-skills install --dest DIR   # DIR/<name>/
    kugelaudio-skills list
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

SKILLS_DIR = Path(__file__).resolve().parent

__all__ = ["SKILLS_DIR", "bundled_skills", "install", "main"]


class SkillInstallError(RuntimeError):
    """Raised when bundled skills are missing or cannot be installed."""


def bundled_skills() -> List[Path]:
    """Return the bundled skill directories (each holding a ``SKILL.md``)."""
    skills = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())
    if not skills:
        raise SkillInstallError(
            f"No skills found in {SKILLS_DIR}. The kugelaudio install looks incomplete; "
            "reinstall the package."
        )
    return skills


def install(dest_root: Path, *, force: bool = False) -> List[Path]:
    """Copy every bundled skill into ``dest_root``; return the paths written."""
    dest_root.mkdir(parents=True, exist_ok=True)
    written = []
    for skill in bundled_skills():
        target = dest_root / skill.name
        if target.exists() and not force:
            print(f"skipped {skill.name} — already at {target} (use --force to overwrite)")
            continue
        shutil.copytree(skill, target, dirs_exist_ok=True)
        print(f"installed {skill.name} -> {target}")
        written.append(target)
    return written


def _resolve_dest(dest: Optional[str], use_global: bool) -> Path:
    if dest:
        return Path(dest).expanduser().resolve()
    if use_global:
        return Path.home() / ".claude" / "skills"
    return Path(".claude/skills").resolve()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kugelaudio-skills",
        description="Install the agent skills bundled with the KugelAudio SDK.",
    )
    parser.add_argument("command", choices=["install", "list"], nargs="?", default="install")
    parser.add_argument("--dest", help="Target skills directory (default: ./.claude/skills)")
    parser.add_argument(
        "--global",
        dest="use_global",
        action="store_true",
        help="Install into ~/.claude/skills instead of the current directory",
    )
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing skills")
    args = parser.parse_args(argv)

    try:
        if args.command == "list":
            for skill in bundled_skills():
                print(skill.name)
            return 0

        dest_root = _resolve_dest(args.dest, args.use_global)
        # A skills directory that did not exist when the agent started is not
        # watched yet, so a brand-new one needs a restart to be picked up.
        dest_root_existed = dest_root.exists()
        written = install(dest_root, force=args.force)
    except SkillInstallError as exc:
        print(f"kugelaudio-skills: {exc}", file=sys.stderr)
        return 1

    if written:
        print(
            "\nClaude Code picks up skills added to a watched skills directory "
            "without a restart."
            if dest_root_existed
            else f"\nCreated {dest_root}. If the skill doesn't show up, restart "
            "Claude Code once so it watches the new directory."
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
