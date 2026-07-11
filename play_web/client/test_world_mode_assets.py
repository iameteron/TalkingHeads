import os
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def test_world_mode_agent_icons_exist():
    craftax = _ASSETS_DIR / "agent-icon-craftax.png"
    exo = _ASSETS_DIR / "agent-icon-exo-planet.png"
    assert craftax.is_file(), f"Missing Craftax UI icon: {craftax}"
    assert exo.is_file(), f"Missing Exo-Planet UI icon: {exo}"
    assert os.path.getsize(craftax) > 0
    assert os.path.getsize(exo) > 0


def test_world_stat_icons_exist():
    stats_dir = _ASSETS_DIR / "game-stats"
    for name in ("health", "food", "drink", "energy"):
        icon = stats_dir / f"{name}.png"
        assert icon.is_file(), f"Missing game stat icon: {icon}"
        assert os.path.getsize(icon) > 0


if __name__ == "__main__":
    test_world_mode_agent_icons_exist()
    test_world_stat_icons_exist()
    print("World mode UI asset tests passed.")
