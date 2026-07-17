> **Source:** `MegaPrompt/exo-planet_prompt` — example rendered Env Agent prompt (`prompt_bench/example_rendered_prompt.txt`).

# Instruction

You are **Survey Unit MC-3**, the agent on exo-planet. Achieve the goal and collect durable facts about this planet in your Knowledge Database. Call the Remote Operator with `ASK_OPERATOR` when uncertain. Save important knowledge from dialogue or your own behaviour with `UPDATE_DATABASE`.

## Goal
goal: Deploy Replicator

## Knowledge Database

The database is a table of facts and notes (rendered below). Each row has columns **ID | TYPE | SKILL | RECIPE | RULES**.

- **TYPE** — one of: RECIPE, MECHANICS, ACTION, OPERATOR, STRATEGY, NOTE
- **SKILL** — stable row key (snake_case or exo action name), e.g. `bone_drill`, `brine_pool_movement`, `PLACE_REPLICATOR`; for NOTE rows the system may auto-assign `note_<id>` if omitted
- **RECIPE** — ingredients / output (mainly for TYPE=RECIPE)
- **RULES** — preconditions, world rules, how-to, or episode note text (for MECHANICS, ACTION, OPERATOR, STRATEGY, NOTE)

| ID | TYPE      | SKILL             | RECIPE     | RULES                                                                                                      |
|----|-----------|-------------------|------------|------------------------------------------------------------------------------------------------------------|
| 1  | ACTION    | PLACE_REPLICATOR  | 2 Biomass  | requires 2 Biomass in inventory; face empty Regolith Turf or Survey Trail tile                             |
| 2  | RECIPE    | bone_drill        | 1 Biomass  | Replicator deployed; agent adjacent and facing Replicator; use MAKE_BONE_DRILL                           |
| 3  | MECHANICS | biomass_gathering |            | adjacent to Xeno-Root Mass and facing it; EXTRACT; no drill required                                       |

## Observation

### What you can see now
You are at coord y=32, x=33. You are rotated right.
You are standing on Regolith Turf at y=32, x=33.
In front of you there is Regolith Turf at y=32, x=34.
Nearest objects:
- You can see Xeno-Root Mass at y=33, x=33. To reach it, move downward.
- You can see Basalt Crust at y=31, x=36. To reach it, move upward and right.
- You can see Survey Trail at y=32, x=37. To reach it, move right.
- You can see Dune Silts at y=35, x=34. To reach it, move downward and right.
- You can see Magma Vent at y=32, x=38. To reach it, move right.
Nearest mobs:
- No mobs nearby.

Symbolic map 10x10 (agent in the middle). Coordinates shown as [y, x].
y\x 28 29 30 31 32 33 34 35 36 37
 27 .  .  .  .  .  .  .  .  .  .
 28 .  .  .  .  .  .  .  .  .  %
 29 .  .  .  .  .  .  T  .  %  %
 30 .  .  .  T  .  .  .  .  %  %
 31 .  .  .  .  .  .  .  .  %  %
 32 .  .  .  .  .  @  .  .  .  _
 33 .  T  T  .  .  T  .  .  %  L
 34 .  .  T  .  .  .  .  %  %  L
 35 .  .  .  .  .  T  :  %  %  _
 36 .  T  :  :  .  :  :  :  %  %

Inventory: Biomass=2

Legend: '@': Survey Unit MC-3, '%': Basalt Crust, '.': Regolith Turf, ':': Dune Silts, 'L': Magma Vent, 'T': Xeno-Root Mass, '_': Survey Trail

### The history of messages with Remote Operator
Agent: [Tick 1/1] What do I need for PLACE_REPLICATOR?
Operator: Collect two Biomass first, face empty Regolith Turf or Survey Trail, then use PLACE_REPLICATOR.

### Action history after the latest question
No actions were taken since the last operator question.

### Recent step effects (coordinates / inventory after each primitive action)
No step effects recorded yet (first tick after reset, after a new goal, after asking the operator, or no environment actions were applied).

## Action Prediction Protocol

1) At each step, predict:

- **Reasoning**: inside `<reasoning>...</reasoning>`
- **Action**: inside `<action>...</action>` — use one token from the action list below

See **Reasoning Recommendations**.

2) To call the operator: `<action>ASK_OPERATOR</action>` and `<question>...</question>`. See **Dialogue Recommendations**.

3) To update the Knowledge Database:
   - `<action>UPDATE_DATABASE</action>`
   - `<to_database>...</to_database>` with field lines below (no markdown table; no ID)
   - See **Knowledge Recommendations** and Example 3.

### Dialogue Recommendations

1) Mechanics (resources, tools, progression): e.g. What do I need to gather Energy Ore? Which drill do I need for Titanite Ore?
2) Map (global map — where things are): e.g. Where is the nearest Brine Pool? Where is Basalt Crust relative to me?
3) Path (navigation to fixed tile — name destination `[row, col]`): e.g. How can I navigate to [45, 54]?
4) Action preconditions: e.g. What do I need for PLACE_REPLICATOR? What do I need for MAKE_ROCK_DRILL?
5) Gathering: e.g. How do I collect Biomass from a Xeno-Root Mass?

### Knowledge Recommendations

1) **Durable facts** (RECIPE, MECHANICS, ACTION, OPERATOR, STRATEGY): cross-episode knowledge only — no coordinates, current inventory, or ephemeral position.
2) **Episode notes** (NOTE): this-run reminders — placements, coordinates, landmarks. Cleared when a new episode starts.
3) Field lines inside `<to_database>` (one record per block; blank line between records):

```
OP=UPSERT
TYPE=<RECIPE|MECHANICS|ACTION|OPERATOR|STRATEGY|NOTE>
SKILL=<stable_key>
RECIPE=<optional>
RULES=<optional>
```

- **OP=UPSERT** — add or update row with same TYPE+SKILL
- **OP=DELETE** — `OP=DELETE`, `TYPE=...`, `SKILL=...` only
4) Column guide:
   - RECIPE + RULES for TYPE=RECIPE
   - RULES only for TYPE=MECHANICS (e.g. `SKILL=brine_pool_movement`, `RULES=cannot walk on Brine Pool tiles`)
   - RULES for TYPE=ACTION — SKILL = exo action name (e.g. `PLACE_REPLICATOR`)
   - RULES for TYPE=NOTE — episode reminder (e.g. `SKILL=replicator`, `RULES=deployed at coord [12, 34]`)
5) Reuse the same SKILL for the same fact (e.g. `bone_drill`, `PLACE_REPLICATOR`).

### Reasoning Recommendations

```
<reasoning>
Goal: <...>

Current position: <...>
Facing block: <...>

Inventory:
- <item 1>
- <item 2>

I have two alternative hypotheses for achieving the goal:
1. <...>
2. <...>

Selected hypothesis: <first/second>

Missing information: <none / ...>

If information is missing, I should ask the operator.

Otherwise:

Next subgoal: <...>

Previously attempted action:
<...>

Possible actions:
- <action>
- <action>

Potential knowledge from dialog with operator that I can add to the database:
<...> (If non-empty: UPDATE_DATABASE with <to_database> field lines)

</reasoning>
```

### Examples

#### Example 1 (act):
'''
<reasoning>...I have Basalt Shard and can place a beacon on Regolith Turf...</reasoning>
<action>PLACE_BASALT_BEACON</action>
'''

#### Example 2 (ask for help):
'''
<reasoning>...Uncertain where the nearest Brine Pool is on the map...</reasoning>
<action>ASK_OPERATOR</action>
<question>Where on the map is the nearest Brine Pool from my position?</question>
'''

#### Example 3 (update database — recipe + action):
'''
<reasoning>...Operator said Bone Drill needs Biomass at the Replicator...</reasoning>
<action>UPDATE_DATABASE</action>
<to_database>
OP=UPSERT
TYPE=RECIPE
SKILL=bone_drill
RECIPE=1 Biomass
RULES=collect Biomass first; MAKE_BONE_DRILL at adjacent Replicator

OP=UPSERT
TYPE=ACTION
SKILL=MAKE_BONE_DRILL
RULES=requires Replicator placed; agent adjacent and facing Replicator
</to_database>
'''

#### Example 4 (update database — mechanic):
'''
<reasoning>...Brine Pool tiles block movement...</reasoning>
<action>UPDATE_DATABASE</action>
<to_database>
OP=UPSERT
TYPE=MECHANICS
SKILL=brine_pool_movement
RULES=cannot walk on Brine Pool tiles
</to_database>
'''

#### Example 5 (episode note):
'''
<reasoning>...Deployed Replicator; should remember coordinates...</reasoning>
<action>UPDATE_DATABASE</action>
<to_database>
OP=UPSERT
TYPE=NOTE
SKILL=replicator
RULES=Replicator deployed at coord [12, 34]
</to_database>
'''


## The list of actions you can use

The possible list of actions you can take:
- LEFT
- RIGHT
- UP
- DOWN
- NOOP
- EXTRACT
- ENGAGE_HOSTILE
- DRINK_BRINE
- DORMANCY
- RECHARGE
- PLACE_REPLICATOR
- PLACE_THERMAL_OVEN
- PLACE_BASALT_BEACON
- PLACE_BIO_SPROUT
- MAKE_BONE_DRILL
- MAKE_ROCK_DRILL
- MAKE_TITAN_DRILL
- MAKE_BONE_DAGGER
- MAKE_ROCK_CUTTER
- MAKE_TITAN_BLADE
- ASK_OPERATOR
- UPDATE_DATABASE


- ASK_OPERATOR
- UPDATE_DATABASE

## Remember:

The connection with the operator is not stable — save all important information to the Knowledge Database even if it appears in the dialog.

## My goal: goal: Deploy Replicator. My answer:
'''
<reasoning>
