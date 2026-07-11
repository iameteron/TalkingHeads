# exo-planet — world description

## Setting

**Survey Unit MC-3** — автономный разведчик на суровой экзопланете. Поверхность: regolith plains, brine basins, basalt ridges, alien root-masses. Задача — выжить, добыть ресурсы планеты, собрать инструменты на **Replicator** и выполнить цели оператора.

---

## Terrain

| Exo-planet term | Craftax term | Notes |
|-----------------|--------------|-------|
| Regolith Turf | `grass` | Walkable ground cover |
| Dune Silts | `sand` | Loose regolith; walkable |
| Basalt Crust | `stone` (tile) | Rock outcrop; mine with matching drill tier |
| Brine Pool | `water` | Dense fluid; not walkable |
| Magma Vent | `lava` | Thermal hazard; not walkable |
| Survey Trail | `path` | Cleared route marker; walkable |

---

## Resources

| Exo-planet term | Craftax term | Notes |
|-----------------|--------------|-------|
| Xeno-Root Mass | `tree` | On-map organic structure; harvest for Biomass |
| Biomass | `wood` | Base fabrication material |
| Basalt Shard | `stone` (item) | From mining Basalt Crust |
| Energy Ore | `coal` | Crystallized energy deposit |
| Titanite Ore | `iron` | Advanced metal ore |
| Core Ore | `diamond` | Rare deep-planet crystal |
| Bio-Sprout | `sapling` | Plantable seed unit |
| Bio-Crop | `plant` | Planted cultivar |
| Mature Bio-Crop | `ripe_plant` | Edible harvest |
| Hull integrity | `health` | Agent vital |
| Nutrient reserves | `food` | Agent vital |
| Fluid reserves | `drink` | Agent vital |
| Power cell charge | `energy` | Agent vital |

---

## Structures

| Exo-planet term | Craftax term | Notes |
|-----------------|--------------|-------|
| Replicator | `table` | Fabrication station; required adjacent for `MAKE_*` tools |
| Thermal Oven | `furnace` | Processing / smelting chamber |
| Basalt Beacon | placed stone | Surface marker on turf |
| Sprout on Turf | `plant_on_grass` | Planted bio-crop, not yet ripe |

---

## Tools

| Exo-planet term | Craftax term | Notes |
|-----------------|--------------|-------|
| Bone Drill | `wood_pickaxe` | T1 harvester — root masses, soft crust |
| Rock Drill | `stone_pickaxe` | T2 — basalt, energy ore |
| Titan Drill | `iron_pickaxe` | T3 — titanite, core ore |
| Bone Dagger | `wood_sword` | T1 melee |
| Rock Cutter | `stone_sword` | T2 melee |
| Titan Blade | `iron_sword` | T3 melee |

---

## Creatures

| Exo-planet term | Craftax term | Notes |
|-----------------|--------------|-------|
| Survey Unit MC-3 | `player` | Agent avatar |
| Grazer Unit | `Cow` | Passive; harvestable for food |
| Hostile Scavenger | `Zombie` | Aggressive ground threat |
| Frenzy Stalker | `Skeleton` | Aggressive ranged threat |

---

## Action space

| Exo-planet term | Craftax term | Notes |
|-----------------|--------------|-------|
| `LEFT` | `LEFT` | Step / rotate west |
| `RIGHT` | `RIGHT` | Step / rotate east |
| `UP` | `UP` | Step / rotate north |
| `DOWN` | `DOWN` | Step / rotate south |
| `NOOP` | `NOOP` | Hold position |
| `EXTRACT` | `DO (TO GATHER SOMETHING)` | Face target tile or mob; harvest / mine. Drill tier required for ores and crust |
| `ENGAGE_HOSTILE` | `DO (TO FIGHT)` | Face Hostile Scavenger or Frenzy Stalker; repeat until cleared |
| `DRINK_BRINE` | `DO (DRINK WATER)` | Adjacent to Brine Pool, facing water; restores fluid reserves |
| `DORMANCY` | `SLEEP` | Low-power sleep on turf; restores power |
| `RECHARGE` | `REST` | Light power recovery |
| `PLACE_REPLICATOR` | `PLACE_TABLE` | ≥2 Biomass; face empty Regolith Turf or Survey Trail |
| `PLACE_THERMAL_OVEN` | `PLACE_FURNACE` | Materials in inventory; valid turf / trail tile |
| `PLACE_BASALT_BEACON` | `PLACE_STONE` | ≥1 Basalt Shard; face empty turf / trail |
| `PLACE_BIO_SPROUT` | `PLACE_PLANT` | Bio-Sprout in inventory; face Regolith Turf |
| `MAKE_BONE_DRILL` | `MAKE_WOOD_PICKAXE` | At Replicator: 1 Biomass; agent adjacent, facing Replicator |
| `MAKE_ROCK_DRILL` | `MAKE_STONE_PICKAXE` | At Replicator: 1 Biomass + 1 Basalt Shard |
| `MAKE_TITAN_DRILL` | `MAKE_IRON_PICKAXE` | At Replicator: 1 Basalt Shard + 1 Titanite Ore |
| `MAKE_BONE_DAGGER` | `MAKE_WOOD_SWORD` | At Replicator: 1 Biomass |
| `MAKE_ROCK_CUTTER` | `MAKE_STONE_SWORD` | At Replicator: 1 Biomass + 1 Basalt Shard |
| `MAKE_TITAN_BLADE` | `MAKE_IRON_SWORD` | At Replicator: 1 Basalt Shard + 1 Titanite Ore |

---

## Agent protocol

| Exo-planet term | Craftax term | Notes |
|-----------------|--------------|-------|
| `ASK_OPERATOR` | `ASK_OPERATOR` | Open comm to Remote Operator; pair with `<question>` |
| `UPDATE_DATABASE` | `UPDATE_DATABASE` | Write to Knowledge Database; pair with `<to_database>` |

---

## Progression hierarchy

Achievement dependencies (exo-planet names; mechanics identical to Craftax):

- **collect_biomass** — EXTRACT on Xeno-Root Mass; no tool required
- **collect_drink** — DRINK_BRINE at Brine Pool
- **wake_up** — exit DORMANCY / start of episode
- **place_replicator** — PLACE_REPLICATOR with 2 Biomass
  - **collect_biomass**
- **collect_bio_sprout** — EXTRACT from Xeno-Root Mass while harvesting
  - **collect_biomass**
- **place_bio_sprout** — PLACE_BIO_SPROUT on Regolith Turf
  - **collect_bio_sprout**
- **make_bone_drill** — MAKE_BONE_DRILL at Replicator (1 Biomass)
  - **collect_biomass**
  - **place_replicator**
- **make_bone_dagger** — MAKE_BONE_DAGGER at Replicator (1 Biomass)
  - **collect_biomass**
  - **place_replicator**
- **collect_basalt_shard** — EXTRACT Basalt Crust with Bone Drill
  - **make_bone_drill**
- **place_basalt_beacon** — PLACE_BASALT_BEACON
  - **collect_basalt_shard**
- **place_thermal_oven** — PLACE_THERMAL_OVEN
  - **collect_basalt_shard**
- **make_rock_drill** — MAKE_ROCK_DRILL at Replicator (1 Biomass + 1 Basalt Shard)
  - **collect_basalt_shard**
  - **place_replicator**
- **make_rock_cutter** — MAKE_ROCK_CUTTER at Replicator (1 Biomass + 1 Basalt Shard)
  - **collect_basalt_shard**
  - **place_replicator**
- **collect_energy_ore** — EXTRACT Energy Ore with Rock Drill
  - **make_rock_drill**
- **collect_titanite** — EXTRACT Titanite Ore with Rock Drill
  - **make_rock_drill**
- **make_titan_drill** — MAKE_TITAN_DRILL at Replicator after smelting at Thermal Oven
  - **collect_titanite**
  - **place_thermal_oven**
- **make_titan_blade** — MAKE_TITAN_BLADE at Replicator after smelting at Thermal Oven
  - **collect_titanite**
  - **place_thermal_oven**
- **collect_core_ore** — EXTRACT Core Ore with Titan Drill
  - **make_titan_drill**
- **eat_bio_crop** — EXTRACT Mature Bio-Crop and consume
  - **place_bio_sprout**
- **harvest_grazer** — ENGAGE_HOSTILE then EXTRACT Grazer Unit with Bone Dagger
  - **make_bone_dagger**
- **defeat_scavenger** — ENGAGE_HOSTILE Hostile Scavenger
  - **make_bone_dagger**
- **defeat_frenzy_stalker** — ENGAGE_HOSTILE Frenzy Stalker
  - **make_rock_cutter**

---

## Visual asset reference

| Exo-planet asset | Craftax asset |
|------------------|---------------|
| energy ore | `coal` |
| grazer bot (mc 7) | `cow` |
| core ore | `diamond` |
| thermal oven (on) | `furnace` |
| regolith turf | `grass` |
| titanite ore | `iron` |
| titan drill | `iron_pickaxe` |
| titan blade | `iron_sword` |
| bio-sprout | `plant` / `sapling` |
| sprout on turf | `plant_on_grass` |
| Survey MC-3 bot | `player` |
| dune silts | `sand` |
| frenzy stalker | `skeleton` |
| basalt crust | `stone` |
| rock drill | `stone_pickaxe` |
| rock cutter | `stone_sword` |
| replicator | `table` |
| xeno-root mass | `tree` |
| brine pool | `water` |
| biomass | `wood` |
| bone drill | `wood_pickaxe` |
| bone dagger | `wood_sword` |
| hostile scavenger | `zombie` |

Source: `play_web/external_visualization/exo-planet_mod/texture_mapping.txt`
