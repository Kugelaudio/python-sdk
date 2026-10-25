"""Tests for the bundled agent skills and the ``kugelaudio-skills`` command."""

from pathlib import Path

import pytest
from kugelaudio.skills import bundled_skills, install, main

# The canonical copies live in the monorepo's packages/private, which the public
# mirror of this package and its sdist do not have.
PRIVATE_DIR = Path(__file__).resolve().parents[3] / "private"
CANONICAL_DIR = PRIVATE_DIR / "tools" / "agent-skills"
SYNC = "packages/private/tools/agent-skills/sync.sh"


def test_every_skill_has_valid_frontmatter():
    for skill in bundled_skills():
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{skill.name}: missing frontmatter"
        frontmatter = text.split("---", 2)[1]
        assert f"name: {skill.name}" in frontmatter, f"{skill.name}: name must match directory"
        assert "description:" in frontmatter, f"{skill.name}: description is required"


def test_install_copies_skills(tmp_path):
    written = install(tmp_path / "skills")
    assert written
    for target in written:
        assert (target / "SKILL.md").is_file()


def test_install_does_not_overwrite_without_force(tmp_path):
    dest = tmp_path / "skills"
    install(dest)
    marker = dest / "kugelaudio-tts" / "SKILL.md"
    marker.write_text("edited by the user", encoding="utf-8")

    install(dest)
    assert marker.read_text(encoding="utf-8") == "edited by the user"

    install(dest, force=True)
    assert marker.read_text(encoding="utf-8") != "edited by the user"


def test_cli_list(capsys):
    assert main(["list"]) == 0
    listed = capsys.readouterr().out.split()
    assert "kugelaudio-tts" in listed
    assert "kugelaudio-agents" in listed


@pytest.mark.skipif(not PRIVATE_DIR.is_dir(), reason="the canonical skills live in packages/private")
def test_bundled_skills_are_exactly_the_canonical_ones():
    assert CANONICAL_DIR.is_dir(), f"no canonical skills at {CANONICAL_DIR}"
    bundled = {skill.name: skill for skill in bundled_skills()}
    canonical = {p.name: p for p in CANONICAL_DIR.iterdir() if (p / "SKILL.md").is_file()}
    assert sorted(bundled) == sorted(canonical), (
        f"the bundled skills differ from the canonical ones; run {SYNC}"
    )
    for name, skill in bundled.items():
        assert (skill / "SKILL.md").read_text(encoding="utf-8") == (
            canonical[name] / "SKILL.md"
        ).read_text(encoding="utf-8"), f"{name} is out of sync; run {SYNC}"
