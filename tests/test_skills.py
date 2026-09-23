"""Tests for the bundled agent skills and the ``kugelaudio-skills`` command."""

from pathlib import Path

import pytest
from kugelaudio.skills import bundled_skills, install, main

# Canonical copies live at the repo root; absent in an installed wheel.
CANONICAL_DIR = Path(__file__).resolve().parents[3] / "agent-skills"


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
    assert "kugelaudio-tts" in capsys.readouterr().out


@pytest.mark.skipif(not CANONICAL_DIR.is_dir(), reason="canonical skills only exist in the repo")
def test_bundled_copy_matches_canonical():
    for skill in bundled_skills():
        canonical = CANONICAL_DIR / skill.name / "SKILL.md"
        assert canonical.is_file(), f"{skill.name} has no canonical source in agent-skills/"
        assert (skill / "SKILL.md").read_text(encoding="utf-8") == canonical.read_text(
            encoding="utf-8"
        ), f"{skill.name} is out of sync — run agent-skills/sync.sh"


def test_no_stale_bundled_skill():
    """A skill deleted from agent-skills/ must not linger in the package."""
    if not CANONICAL_DIR.is_dir():
        pytest.skip("canonical skills only exist in the repo")
    bundled = {p.name for p in bundled_skills()}
    canonical = {p.name for p in CANONICAL_DIR.iterdir() if (p / "SKILL.md").is_file()}
    assert bundled <= canonical, f"stale skills in the package: {sorted(bundled - canonical)}"
