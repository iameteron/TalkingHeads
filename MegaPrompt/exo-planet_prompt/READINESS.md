# exo-planet agent prompts — readiness checklist

## Prompt stack (go)

- [x] `exo_database_formulation` — Knowledge Database + `UPDATE_DATABASE`
- [x] `exo_reasoning_or_ask_path` — navigation + `ASK_OPERATOR`
- [x] `exo_reasoning_or_ask_help` — goal help + `ASK_OPERATOR`
- [x] `exo_no_dialog` — solo navigation
- [x] World lore in `world/exo_planet_world.md` injected via renderer
- [x] exo action bridge → Craftax in `play_web` (`action_bridge.py`)

## Contract validation (go)

- [x] `prompt_bench/test_contract.py` — render all configs, action bridge, legacy guards
- [x] `prompt_bench/llm.py` — exo action-space + mixed-output guards

## Benchmarks (run before experiments)

```bash
cd MegaPrompt
python3 -m unittest exo-planet_prompt/prompt_bench/test_contract.py

# Optional — requires local HF model + GPU:
python3 exo-planet_prompt/prompt_bench/test_sr.py \
  --render-config exo-planet_prompt/templates/database_formulation/exo-planet.yaml
python3 exo-planet_prompt/prompt_bench/test_ask_rate.py \
  --render-config exo-planet_prompt/templates/reasoning_or_ask_path/exo-planet.yaml
```

## play_web e2e smoke (manual)

1. Set `megaprompt_config_name` to `exo_database_formulation` (or other `exo_*` config).
2. Run agent tick with operator loop on exo-planet textures.
3. Verify: parseable `<action>` / `<question>`, exo terms in output, env accepts mapped actions.

## go/no-go for `exo-repeat-craftext-exp`

| Criterion | Target |
|-----------|--------|
| `test_contract.py` | pass |
| Valid tick rate (e2e) | ≥ 90% parseable steps |
| Legacy terms in agent output | 0 |
| Operator questions | exo vocabulary, routable |
