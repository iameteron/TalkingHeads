# ARC-AGI-3 prompt stack

Prompt stack for ARC-AGI-3 games integrated in TalkingHeads.

## Configs

| Config name | Observation format |
|-------------|--------------------|
| `arc_2_image` | Text metadata plus previous and current PNG frame images |
| `arc_grid` | Text metadata plus a 64x64 hexadecimal palette grid |
| `arc_grid_image` | Text metadata plus both the 64x64 grid and an attached PNG frame image |
| `arc_image` | Text metadata plus an attached PNG frame image |

The image-bearing configs emit one or more `[[image:data:image/png;base64,...]]`
markers.
TalkingHeads' OpenRouter active-agent transport converts that marker into a
multimodal `image_url` chat content item before sending it to the model.

## Shared Prompt Sections

All ARC configs use the same task framing and output contract. The rendered
prompt includes:

- current goal;
- current step number;
- available actions;
- current observation;
- messages from the human operator;
- recent action history;
- previous agent reasoning;
- final output rules.

The action descriptions intentionally use the visible control semantics:

```text
ACTION1      Up arrow
ACTION2      Down arrow
ACTION3      Left arrow
ACTION4      Right arrow
ACTION5      Spacebar / interact / select when available
ACTION6 x y  Click on the game frame
ACTION7      Undo when available
```

For click actions, the final action block must use `ACTION6 x y` with `0..63`
frame coordinates, for example:

```text
--- Act ---
ACTION6 32 31
--- Act ---
```

The prompt tells the agent to minimize environment actions and to ask the human
operator when it is uncertain about game rules, the objective, or a repeated
failure pattern.

In the `dev` app profile, the Settings modal can append custom ARC prompt text.
In the `demo` profile, that override is hidden and ignored by the backend.
