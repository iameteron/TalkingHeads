"""exo-planet achievement dependency graph for mechanics expert."""

exo_achievement_dependencies = {
    "collect_biomass": {
        "deps": [],
        "info": "EXTRACT on a Xeno-Root Mass; no drill required.",
    },
    "collect_drink": {
        "deps": [],
        "info": "DRINK_BRINE while adjacent to a Brine Pool and facing it.",
    },
    "wake_up": {
        "deps": [],
        "info": "Exit DORMANCY or start of episode.",
    },
    "place_replicator": {
        "deps": ["collect_biomass"],
        "info": "PLACE_REPLICATOR with 2 Biomass on empty Regolith Turf or Survey Trail.",
    },
    "collect_bio_sprout": {
        "deps": ["collect_biomass"],
        "info": "EXTRACT from Xeno-Root Mass; Bio-Sprout may drop while harvesting.",
    },
    "place_bio_sprout": {
        "deps": ["collect_bio_sprout"],
        "info": "PLACE_BIO_SPROUT on Regolith Turf.",
    },
    "make_bone_drill": {
        "deps": ["collect_biomass", "place_replicator"],
        "info": "MAKE_BONE_DRILL at Replicator (1 Biomass); agent adjacent and facing Replicator.",
    },
    "make_bone_dagger": {
        "deps": ["collect_biomass", "place_replicator"],
        "info": "MAKE_BONE_DAGGER at Replicator (1 Biomass).",
    },
    "collect_basalt_shard": {
        "deps": ["make_bone_drill"],
        "info": "EXTRACT Basalt Crust with Bone Drill equipped.",
    },
    "place_basalt_beacon": {
        "deps": ["collect_basalt_shard"],
        "info": "PLACE_BASALT_BEACON with at least 1 Basalt Shard.",
    },
    "place_thermal_oven": {
        "deps": ["collect_basalt_shard"],
        "info": "PLACE_THERMAL_OVEN on valid turf or trail tile.",
    },
    "make_rock_drill": {
        "deps": ["collect_basalt_shard", "place_replicator"],
        "info": "MAKE_ROCK_DRILL at Replicator (1 Biomass + 1 Basalt Shard).",
    },
    "make_rock_cutter": {
        "deps": ["collect_basalt_shard", "place_replicator"],
        "info": "MAKE_ROCK_CUTTER at Replicator (1 Biomass + 1 Basalt Shard).",
    },
    "collect_energy_ore": {
        "deps": ["make_rock_drill"],
        "info": "EXTRACT Energy Ore with Rock Drill.",
    },
    "collect_titanite": {
        "deps": ["make_rock_drill"],
        "info": "EXTRACT Titanite Ore with Rock Drill.",
    },
    "make_titan_drill": {
        "deps": ["collect_titanite", "place_thermal_oven"],
        "info": "MAKE_TITAN_DRILL at Replicator after smelting at Thermal Oven.",
    },
    "make_titan_blade": {
        "deps": ["collect_titanite", "place_thermal_oven"],
        "info": "MAKE_TITAN_BLADE at Replicator after smelting at Thermal Oven.",
    },
    "collect_core_ore": {
        "deps": ["make_titan_drill"],
        "info": "EXTRACT Core Ore with Titan Drill.",
    },
    "eat_bio_crop": {
        "deps": ["place_bio_sprout"],
        "info": "EXTRACT Mature Bio-Crop and consume.",
    },
    "harvest_grazer": {
        "deps": ["make_bone_dagger"],
        "info": "ENGAGE_HOSTILE then EXTRACT Grazer Unit with Bone Dagger.",
    },
    "defeat_scavenger": {
        "deps": ["make_bone_dagger"],
        "info": "ENGAGE_HOSTILE against Hostile Scavenger until cleared.",
    },
    "defeat_frenzy_stalker": {
        "deps": ["make_rock_cutter"],
        "info": "ENGAGE_HOSTILE against Frenzy Stalker until cleared.",
    },
}
