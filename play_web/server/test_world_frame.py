import numpy as np

from play_web.server.world_frame import _inventory_items, _inventory_payload


class _FakeInv:
    wood = 2
    stone = 0
    coal = 1
    iron = 0
    diamond = 0
    sapling = 0
    wood_pickaxe = 0
    stone_pickaxe = 0
    iron_pickaxe = 0
    wood_sword = 0
    stone_sword = 0
    iron_sword = 0


class _FakeState:
    inventory = _FakeInv()


def test_inventory_payload_and_items():
    payload = _inventory_payload(_FakeState())
    assert payload["wood"] == 2
    assert payload["stone"] == 0
    items = _inventory_items(payload)
    keys = {item["key"] for item in items}
    assert keys == {"wood", "coal"}
    assert all(item["count"] > 0 for item in items)


def test_map_diff_selection():
    old = np.array([[1, 2], [3, 4]], dtype=np.int32)
    new = np.array([[1, 9], [3, 4]], dtype=np.int32)
    mask = new != old
    xs, ys = np.where(mask)
    assert list(zip(xs.tolist(), ys.tolist())) == [(0, 1)]
