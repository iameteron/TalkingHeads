from __future__ import annotations

from typing import Any

from megaprompt.vocabulary.context import exo_planet_vocabulary, use_vocabulary

from ._shared import render_map_and_coords_text


def render(state: Any) -> str:
    with use_vocabulary(exo_planet_vocabulary()):
        return render_map_and_coords_text(state)
