# ARC-AGI-3 Integration

TalkingHeads supports a lightweight ARC-AGI-3 world mode for interactive local
experiments. The current integration targets fast iteration in the web UI, not
official online scorecard/replay submission.

## Supported Games

The supported games are:

- `ar25`
- `bp35`
- `ls20`
- `lp85`

Game files are expected under:

```text
play_web/environment_files/<game_id>/<environment_hash>/
```

The backend uses the official `arc-agi` SDK in offline mode:

```python
Arcade(operation_mode=OperationMode.OFFLINE)
```

If `arc-agi` is missing or a local game file cannot be loaded, ARC mode returns
a controlled error instead of breaking Craftax startup.

## Runtime Behavior

ARC mode is selected with:

```text
game_kind = "arc_agi"
arc_game_id = "ar25" | "bp35" | "ls20" | "lp85"
```

The default world remains Craftax. Switching to ARC:

- resets the environment through `ArcAgiAdapter`;
- forces `interaction_mode = "human"`;
- disables Craftax oracle experts;
- disables companion/campaign flows;
- uses ARC-specific prompt generation instead of Craftax/Exo MegaPrompt paths.

This is intentional: ARC games currently have no bespoke oracle or expert
system in TalkingHeads.

## Observation Formats

ARC prompts are implemented as a separate MegaPrompt family.

| Config | What the agent receives |
|--------|--------------------------|
| `arc_2_image` | Text metadata plus previous and current frame images for comparing the latest change. |
| `arc_grid` | Text metadata plus a 64x64 hexadecimal palette grid. |
| `arc_grid_image` | Text metadata plus both the 64x64 palette grid and the rendered game frame as an image marker. |
| `arc_image` | Text metadata plus the rendered game frame as an image marker. |

`arc_image`, `arc_grid_image`, and `arc_2_image` emit image markers:

```text
[[image:data:image/png;base64,...]]
```

Before the OpenRouter request, TalkingHeads converts that marker into a
multimodal `image_url` chat content item. Use image-bearing ARC configs only
with a vision-capable OpenRouter model. Text-only models may ignore the image.

The prompt/debug button in the top bar opens the last agent prompt. This is the
recommended way to verify whether the current run used `arc_grid` text,
`arc_image`, the combined `arc_grid_image` marker, or the two-frame
`arc_2_image` marker set.

In `demo` app profile, observation-format selection and the prompt/debug modal
are hidden. The server uses `TALKINGHEADS_DEMO_ARC_OBS_FORMAT`, which defaults
to `arc_image`.

## Agent Prompt Context

The ARC agent prompt is intentionally separate from Craftax/Exo prompts. It
contains:

- the current goal text from the UI;
- the current agent step number;
- available action descriptions;
- the current ARC observation in the selected format;
- persistent human operator Q/A history, including step labels when available;
- recent action history, up to the last 30 actions;
- the agent's reasoning from the previous step;
- optional ARC prompt override text in dev mode.

The agent sees the current ARC frame observation, not a full visual replay of
all previous frames. Long-running memory is provided through operator Q/A,
recent action history, and the previous-step reasoning section. `arc_2_image`
is the exception for visual comparison: it attaches the previous and current
frame images to help the model infer what changed after the latest action.

The prompt asks the model to minimize environment actions because action count
affects the human-helper score. If the model is unsure about the game rules,
objective, or a likely loop, it is encouraged to ask the human operator instead
of spending extra actions on blind trial and error.

## Actions

ARC manual and model actions use the SDK action names:

```text
ACTION1      Up arrow
ACTION2      Down arrow
ACTION3      Left arrow
ACTION4      Right arrow
ACTION5      Spacebar / interact / select when available
ACTION6 x y  Click on the game frame
ACTION7      Undo when available
```

`ACTION6` requires coordinates in the `0..63` range. The agent should write it
as `ACTION6 32 31` or `ACTION6 [32,31]`. In the UI, click the ARC frame to
select coordinates, then press `ACTION6`.

The WebSocket `step` message accepts either:

```json
{"type": "step", "action": "ACTION1"}
```

or:

```json
{"type": "step", "action": "ACTION6", "x": 32, "y": 31}
```

## Agent Questions And Human Helper

When the ARC agent asks a question, TalkingHeads pauses the current agent tick
and displays the pending question in the right-side human helper panel.

After the human answers:

1. The answer is appended to the ARC dialog history.
2. The paused agent tick resumes.
3. No Craftax oracle expert is called.

The ARC prompt includes recent operator Q/A history so the model can use human
answers on later steps.

Operator Q/A history is reset when the ARC episode resets. In the current ARC
demo flow, completing the first level ends the scored attempt and resets the
episode instead of advancing into a multi-level progression UI.

## Human Object Hints

In ARC mode, clicking the rendered game frame opens the same kind of local
information card used by Craftax/Exo tile inspection. The card shows the ARC
coordinate and an English hint for visible objects. `ls20` has specific hints
for the player, white cross, current shape, target shape, timer refill rings,
and maze cells. `lp85` has hints for the visible click-puzzle markers. `ar25`
has hints for its level timer, movable black-and-white object, synchronized gray
object, yellow target object, center divider, and board areas.
`bp35` has hints for the player marker, corridors, breakable green blocks,
walls, and wall details. Unknown ARC objects still fall back to generic
color/object hints until game-specific rules are documented.

## Human-Helper Leaderboard

ARC mode includes a local leaderboard for human-assisted runs. When a game ends
with `WIN` or `GAME_OVER`, the UI opens an arcade-style score modal:

1. The current score is shown.
2. The human can enter a name.
3. The score is appended to the local leaderboard.

Scores are stored as JSONL:

```text
play_web/leaderboard/arc_human.jsonl
```

On deployed instances this path can be redirected with:

```text
PLAY_WEB_LEADERBOARD_DIR
```

### Score Formula

The current local score is intentionally simple:

```text
base
+ levels_completed * 500
- total_actions * 40
- agent_questions * 200
- human_answers * 100
- manual_actions * 250
- elapsed_seconds
```

`base` is:

- `10000` for `WIN`;
- `1000` otherwise.

Manual actions are penalized more heavily than autonomous agent actions because
the leaderboard is meant to score human help to the agent, not direct human
playthroughs.

## API Endpoints

The ARC leaderboard uses:

```text
GET  /api/arc_human_score
POST /api/arc_human_score
GET  /api/arc_human_leaderboard
```

`POST /api/arc_human_score` accepts:

```json
{"player_name": "AAA"}
```

The server rejects duplicate submissions for the same attempt and rejects
submissions before the game reaches `WIN` or `GAME_OVER`.

## Tests

Focused tests:

```bash
python -m pytest \
  oracle/prompts/test_arc_prompt_generation.py \
  play_web/server/test_arc_agi_adapter.py \
  play_web/server/test_arc_agent_parser.py
```

Useful backend syntax check:

```bash
PYTHONPYCACHEPREFIX=/tmp/talkingheads_pycache \
python -m py_compile \
  play_web/server/runtime.py \
  play_web/server/services.py \
  play_web/server/app.py \
  play_web/server/leaderboard.py
```

## Troubleshooting

### The ARC frame does not render

Check that the local game files exist under `play_web/environment_files/` and
that `arc-agi` is installed from `requirements.txt`.

### The model keeps acting instead of describing the frame

The normal ARC agent prompt asks the model to solve the game and return exactly
one action or one question. The goal field is not a free-form chat prompt. Use
the prompt/debug modal to inspect the exact agent input, or use direct agent
chat when you need exploratory model responses.

### `arc_image` does not affect the model

Use a vision-capable OpenRouter model. Hub/text-only modes receive the text
portion but may ignore the image marker.

The Settings modal includes a small ActiveAgent preset dropdown for ARC image
debugging. Presets are only shortcuts for OpenRouter model ids; the free-form
model-id input remains the source of truth and can be edited manually.
