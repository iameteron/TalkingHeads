# exo-planet — world description

## Setting

**Survey Unit MC-3** — автономный разведчик на суровой экзопланете. Поверхность: regolith plains, brine basins, basalt ridges, alien root-masses. Задача — выжить, добыть ресурсы планеты, собрать инструменты на **Replicator** и выполнить цели оператора.

---

## Terrain

| Term | Notes |
|------|-------|
| Regolith Turf | Walkable ground cover |
| Dune Silts | Loose regolith; walkable |
| Basalt Crust | Rock outcrop; mine with matching drill tier |
| Brine Pool | Dense fluid; not walkable |
| Magma Vent | Thermal hazard; not walkable |
| Survey Trail | Cleared route marker; walkable |

---

## Resources

| Term | Notes |
|------|-------|
| Xeno-Root Mass | On-map organic structure; harvest for Biomass |
| Biomass | Base fabrication material |
| Basalt Shard | From mining Basalt Crust |
| Energy Ore | Crystallized energy deposit |
| Titanite Ore | Advanced metal ore |
| Core Ore | Rare deep-planet crystal |
| Bio-Sprout | Plantable seed unit |
| Bio-Crop | Planted cultivar |
| Mature Bio-Crop | Edible harvest |
| Hull integrity | Agent vital |
| Nutrient reserves | Agent vital |
| Fluid reserves | Agent vital |
| Power cell charge | Agent vital |

---

## Structures

| Term | Notes |
|------|-------|
| Replicator | Fabrication station; required adjacent for `MAKE_*` tools |
| Thermal Oven | Processing / smelting chamber |
| Basalt Beacon | Surface marker on turf |
| Sprout on Turf | Planted bio-crop, not yet ripe |

---

## Tools

| Term | Notes |
|------|-------|
| Bone Drill | T1 harvester — root masses, soft crust |
| Rock Drill | T2 — basalt, energy ore |
| Titan Drill | T3 — titanite, core ore |
| Bone Dagger | T1 melee |
| Rock Cutter | T2 melee |
| Titan Blade | T3 melee |

---

## Creatures

| Term | Notes |
|------|-------|
| Survey Unit MC-3 | Agent avatar |
| Grazer Unit | Passive; harvestable for food |
| Hostile Scavenger | Aggressive ground threat |
| Frenzy Stalker | Aggressive ranged threat |

---

## Action space

| Action | Notes |
|--------|-------|
| `LEFT` | Step / rotate west |
| `RIGHT` | Step / rotate east |
| `UP` | Step / rotate north |
| `DOWN` | Step / rotate south |
| `NOOP` | Hold position |
| `EXTRACT` | Face target tile or mob; harvest / mine. Drill tier required for ores and crust |
| `ENGAGE_HOSTILE` | Face Hostile Scavenger or Frenzy Stalker; repeat until cleared |
| `DRINK_BRINE` | Adjacent to Brine Pool, facing water; restores fluid reserves |
| `DORMANCY` | Low-power sleep on turf; restores power |
| `RECHARGE` | Light power recovery |
| `PLACE_REPLICATOR` | ≥2 Biomass; face empty Regolith Turf or Survey Trail |
| `PLACE_THERMAL_OVEN` | Materials in inventory; valid turf / trail tile |
| `PLACE_BASALT_BEACON` | ≥1 Basalt Shard; face empty turf / trail |
| `PLACE_BIO_SPROUT` | Bio-Sprout in inventory; face Regolith Turf |
| `MAKE_BONE_DRILL` | At Replicator: 1 Biomass; agent adjacent, facing Replicator |
| `MAKE_ROCK_DRILL` | At Replicator: 1 Biomass + 1 Basalt Shard |
| `MAKE_TITAN_DRILL` | At Replicator: 1 Basalt Shard + 1 Titanite Ore |
| `MAKE_BONE_DAGGER` | At Replicator: 1 Biomass |
| `MAKE_ROCK_CUTTER` | At Replicator: 1 Biomass + 1 Basalt Shard |
| `MAKE_TITAN_BLADE` | At Replicator: 1 Basalt Shard + 1 Titanite Ore |

---

## Agent protocol

| Action | Notes |
|--------|-------|
| `ASK_OPERATOR` | Open comm to Remote Operator; pair with `<question>` |
| `UPDATE_DATABASE` | Write to Knowledge Database; pair with `<to_database>` |

---

## Progression hierarchy

Achievement dependencies on exo-planet:

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
