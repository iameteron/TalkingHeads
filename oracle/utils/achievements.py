achievement_dependencies = {
    "collect_wood": {
        "deps": [],
        "info": "You can collect wood from trees without any additional tools.",
    },
    "collect_drink": {
        "deps": [],
        "info": "Find a water source and drink from it.",
    },
    "wake_up": {
        "deps": [],
        "info": "Wake up (typically happens at the start / after sleeping).",
    },
    "place_table": {
        "deps": ["collect_wood"],
        "info": "Craft and place a crafting table using 2 collected wood.",
    },
    "place_plant": {
        "deps": ["collect_sapling"],
        "info": "Plant a sapling on suitable ground.",
    },
    "place_stone": {
        "deps": ["collect_stone"],
        "info": "Place stone blocks after you obtain stone.",
    },
    "place_furnace": {
        "deps": ["collect_stone"],
        "info": "Craft and place a furnace using stone.",
    },
    "collect_sapling": {
        "deps": ["collect_wood"],
        "info": "Collect a sapling while chopping / interacting with trees.",
    },
    "collect_stone": {
        "deps": ["make_wood_pickaxe"],
        "info": "Mine stone blocks using a wooden pickaxe.",
    },
    "collect_coal": {
        "deps": ["make_stone_pickaxe"],
        "info": "Mine coal ore using a stone pickaxe.",
    },
    "collect_iron": {
        "deps": ["make_stone_pickaxe"],
        "info": "Mine iron ore using a stone pickaxe.",
    },
    "collect_diamond": {
        "deps": ["make_iron_pickaxe"],
        "info": "Mine diamond ore using an iron pickaxe.",
    },
    "make_wood_pickaxe": {
        "deps": ["collect_wood", "place_table"],
        "info": "Use the crafting table to craft a wooden pickaxe.",
    },
    "make_wood_sword": {
        "deps": ["collect_wood", "place_table"],
        "info": "Use the crafting table to craft a wooden sword.",
    },
    "make_stone_pickaxe": {
        "deps": ["collect_stone", "place_table"],
        "info": "Use the crafting table to craft a stone pickaxe.",
    },
    "make_stone_sword": {
        "deps": ["collect_stone", "place_table"],
        "info": "Use the crafting table to craft a stone sword.",
    },
    "make_iron_pickaxe": {
        "deps": ["collect_iron", "place_furnace"],
        "info": "Smelt iron (via furnace) and craft an iron pickaxe.",
    },
    "make_iron_sword": {
        "deps": ["collect_iron", "place_furnace"],
        "info": "Smelt iron (via furnace) and craft an iron sword.",
    },
    "eat_plant": {
        "deps": ["place_plant"],
        "info": "Harvest the plant and eat it.",
    },
    "eat_cow": {
        "deps": ["make_wood_sword"],
        "info": "Kill a cow with a sword and eat the obtained food.",
    },
    "defeat_zombie": {
        "deps": ["make_wood_sword"],
        "info": "Fight and defeat a zombie (a sword helps).",
    },
    "defeat_skeleton": {
        "deps": ["make_stone_sword"],
        "info": "Fight and defeat a skeleton (stone sword recommended).",
    },
}
