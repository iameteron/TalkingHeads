# exo-planet prompt stack

Prompt stack для среды **exo-planet** — reskin Craftax Classic с exo-лором и action vocabulary.

## Configs (play_web / MegaPrompt)

| Config name | Template |
|-------------|----------|
| `exo_database_formulation` | database + Knowledge DB |
| `exo_reasoning_or_ask_path` | navigation + operator |
| `exo_reasoning_or_ask_help` | goal help + operator |
| `exo_no_dialog` | solo agent |

```python
Renderer(config_path="MegaPrompt/exo-planet_prompt/templates/database_formulation/exo-planet.yaml")
```

## Structure

```
exo-planet_prompt/
  world/exo_planet_world.md
  actions.py / action_bridge.py
  knowledge_data.json
  templates/{database_formulation,reasoning_or_ask_path,reasoning_or_ask_help,no_dialog}/
  prompt_bench/
  READINESS.md
```

## Tests

```bash
cd MegaPrompt
# contract validation (no torch):
python3 -m unittest discover -s exo-planet_prompt/prompt_bench -p 'test_contract.py'
```

## Status

| Component | Status |
|-----------|--------|
| World lore | done |
| All 4 prompt templates | done |
| Action bridge (exo → Craftax) | done |
| Contract unit tests | done |
| LLM benchmarks (SR / ask rate) | ready to run |
| play_web e2e | manual smoke (see READINESS.md) |
