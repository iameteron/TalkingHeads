from server.inventory_icons import _ICON_CACHE, get_inventory_icons


def test_inventory_icons_differ_by_world_mode():
    _ICON_CACHE.clear()
    craftax = get_inventory_icons("craftax")
    exo = get_inventory_icons("exo-planet")
    assert craftax["wood_pickaxe"] != exo["wood_pickaxe"]
    assert craftax["wood"] != exo["wood"]


def test_inventory_icons_cover_hud_slots():
    _ICON_CACHE.clear()
    icons = get_inventory_icons("craftax")
    from server.inventory_icons import INVENTORY_SLOT_ORDER

    for key in INVENTORY_SLOT_ORDER:
        assert key in icons
        assert icons[key].startswith("data:image/png;base64,")
