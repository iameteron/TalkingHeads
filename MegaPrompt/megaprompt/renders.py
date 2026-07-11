from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_TEMPLATES_ROOT = Path(__file__).resolve().parents[1] / "craftext_prompt" / "templates"


def resolve_config_path(
    *,
    config_name: Optional[str] = None,
    config_path: Optional[str | Path] = None,
) -> Path:
    """
    Resolve config path from one of:
    - config_name: <MegaPrompts>/craftext_prompt/templates/<config_name>/craftext.yaml
    - config_path: explicit yaml path
    """
    if bool(config_name) == bool(config_path):
        raise ValueError("Provide exactly one of `config_name` or `config_path`.")

    if config_name:
        resolved = (DEFAULT_TEMPLATES_ROOT / str(config_name) / "craftext.yaml").expanduser().resolve()
    else:
        resolved = Path(config_path).expanduser().resolve()  # type: ignore[arg-type]

    if not resolved.exists():
        raise FileNotFoundError(f"Prompt config not found: {resolved}")
    return resolved


def _render_generic(render_type, value, subdir, function_name, skip_modules=None):
    base_dir = Path(__file__).parent / subdir
    skip = {"__init__"}
    if skip_modules:
        skip.update(skip_modules)

    for file_path in base_dir.glob("*.py"):
        module_name = file_path.stem
        if module_name in skip:
            continue

        if module_name == render_type:
            module = import_module(f"{__package__}.{subdir}.{module_name}")
            if not hasattr(module, function_name):
                raise ValueError(f"Render module '{module_name}' has no {function_name}")
            return getattr(module, function_name)(value)

    raise ValueError(f"Invalid render type: {render_type}")


class Renderer:
    def __init__(
        self,
        config: Optional[str | Path] = None,
        *,
        config_name: Optional[str] = None,
        config_path: Optional[str | Path] = None,
    ):
        if config is not None:
            if config_name is not None or config_path is not None:
                raise ValueError("`config` cannot be used with `config_name`/`config_path`.")
            # Backward compatibility: positional config treated as explicit path.
            self.config_path = resolve_config_path(config_path=config)
        else:
            self.config_path = resolve_config_path(config_name=config_name, config_path=config_path)

        self.config = self.read_config(self.config_path)
        self.protocol = self._extract_protocol(self.config)
        self.renders_config = self.protocol.get("renders", {})
        self.template = self.read_template(self.protocol["template"])

    def read_config(self, config_path: str | Path) -> dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError("Config must be a YAML object")
        return data

    def _extract_protocol(self, config_data: dict[str, Any]) -> dict[str, Any]:
        protocol_items = config_data.get("protocol")
        if not isinstance(protocol_items, list) or not protocol_items:
            raise ValueError("Config must contain non-empty 'protocol' list")
        protocol = protocol_items[0]
        if not isinstance(protocol, dict):
            raise ValueError("Protocol entry must be an object")
        if "template" not in protocol:
            raise ValueError("Protocol entry must define 'template'")
        return protocol

    def read_template(self, template: str) -> str:
        template_path = Path(template)
        if not template_path.is_absolute():
            template_path = self.config_path.parent / template_path
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()

    def render(self, meta_info: dict[str, Any]) -> str:
        renders = self.renders_config
        output = self.template

        for key, spec in renders.items():
            if not isinstance(spec, dict):
                continue

            renderer_name = spec.get("renderer")
            placeholder = spec.get("placeholder")
            if not renderer_name or not placeholder:
                continue

            subdir, function_name = key, "render"
            input_key = key

            if input_key not in meta_info:
                raise KeyError(
                    f"Missing '{input_key}' in meta_info for render key '{key}'"
                )

            rendered = _render_generic(
                renderer_name,
                meta_info[input_key],
                subdir,
                function_name,
            )
            output = output.replace(placeholder, rendered)

        return output
