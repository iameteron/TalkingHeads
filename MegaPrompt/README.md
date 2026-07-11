# MegaPrompts

Prompt rendering and benchmark utilities for CraftText-style agent prompts.

## What is in this folder

- `craftext_prompt/templates/` - Prompt templates (`.txt`) and YAML render configs (`craftext.yaml`).
- `megaprompt/` - Renderer modules used by template placeholders.
- `craftext_prompt/prompt_bench/test_sr.py` - Main prompt benchmark/test entrypoint.
- `craftext_prompt/prompt_bench/llm.py` - Local HF model inference and action extraction helpers.

## Quick setup

Use Python 3.10+ (recommended) and install the required packages:

```bash
cd MegaPrompts
python3 -m venv .venv
source .venv/bin/activate
pip install pyyaml torch transformers
```

Optional: set model through env var:

```bash
export CRAFTEXT_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

## How to run `test_st`

There is no `test_st.py` in this project. The benchmark script is `test_sr.py`.

Run from `MegaPrompts/`:

```bash
python3 craftext_prompt/prompt_bench/test_sr.py
```

Useful flags:

```bash
python3 craftext_prompt/prompt_bench/test_sr.py \
  --render-config craftext_prompt/templates/dialog/craftext.yaml \
  --step-idx -1 \
  --run-count 10 \
  --llm-model Qwen/Qwen2.5-7B-Instruct \
  --llm-max-new-tokens 64 \
  --llm-temperature 1.0 \
  --llm-top-p 0.9
```

What it does:

1. Loads a saved trajectory step from `prompt_bench/extra_files/place_a_table_trajectory.pkl`.
2. Renders prompt text with `Renderer(...)`.
3. Calls the LLM multiple times.
4. Reports success rate and reasoning-tag stats.

## How to add a new template

Template configs are YAML files with a single `protocol` entry. The renderer engine expects:

- `template` -> path to a `.txt` file with placeholders.
- `renders.<key>.placeholder` -> exact placeholder text in the template.
- `renders.<key>.renderer` -> python module name under `megaprompt/<key>/`.

### 1) Create a new template folder

Example:

```bash
mkdir -p craftext_prompt/templates/my_template
```

### 2) Add template text

Create `craftext_prompt/templates/my_template/reasoning.txt`:

```txt
# Instruction
Goal:
{{goal}}

Observation:
{{observation}}

Dialog:
{{dialog}}

Actions:
{{action}}

<reasoning>...</reasoning>
<action>...</action>
```

### 3) Add YAML config

Create `craftext_prompt/templates/my_template/craftext.yaml`:

```yaml
protocol:
  - name: my_prompt_v1
    template: reasoning.txt
    renders:
      act:
        placeholder: "{{action}}"
        renderer: bullet_list
      obs:
        placeholder: "{{observation}}"
        renderer: map_and_coords
      dialog:
        placeholder: "{{dialog}}"
        renderer: last_five
      goal:
        placeholder: "{{goal}}"
        renderer: just_a_goal
```

### 4) Reuse or add renderers

You can reuse existing renderer modules:

- `act/bullet_list.py`
- `obs/map_and_coords.py`
- `obs/balrog_text.py`
- `dialog/last_five.py`
- `goal/just_a_goal.py`

If you need a custom renderer:

1. Put a file in the matching key folder, for example:
   - key `obs` -> `megaprompt/obs/my_obs_renderer.py`
2. Implement a `render(value)` function in that file.
3. In YAML, set:
   - `renders.obs.renderer: my_obs_renderer`

Important: the `renders` key name maps directly to subfolders under `megaprompt/`:

- `act` -> `megaprompt/act/`
- `obs` -> `megaprompt/obs/`
- `dialog` -> `megaprompt/dialog/`
- `goal` -> `megaprompt/goal/`

### 5) Test the new template

```bash
python3 craftext_prompt/prompt_bench/test_sr.py \
  --render-config craftext_prompt/templates/my_template/craftext.yaml
```

If rendering fails, check:

- Placeholder names match exactly between `.txt` and YAML.
- Renderer module file exists in the correct subfolder.
- Renderer file exports `render(...)`.

