from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEGAPROMPT_ROOT = _REPO_ROOT / "MegaPrompt"
CRAFTEXT_PROMPT_ROOT = _MEGAPROMPT_ROOT / "craftext_prompt"
KNOWLEDGE_JSON_PATH = CRAFTEXT_PROMPT_ROOT / "knowledge_data.json"
KNOWLEDGE_DATA_PATH = CRAFTEXT_PROMPT_ROOT / "knowledge_data.txt"
DATABASE_FORMULATION_DIR = CRAFTEXT_PROMPT_ROOT / "templates" / "database_formulation"
REASONING_TEMPLATE_PATH = DATABASE_FORMULATION_DIR / "reasoning.txt"

_EMPTY_KNOWLEDGE_PLACEHOLDER = (
    "(none yet — facts will accumulate from <to_database> tags)"
)

_KNOWLEDGE_TYPES = frozenset(
    {"RECIPE", "MECHANICS", "ACTION", "OPERATOR", "STRATEGY", "NOTE"}
)
_LEGACY_PREFIXES = _KNOWLEDGE_TYPES | {"CORRECTION", "UPDATE"}

_TO_DATABASE_PATTERN = re.compile(
    r"<to_database>\s*(.*?)\s*</to_database>",
    flags=re.IGNORECASE | re.DOTALL,
)
_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")

_RUNTIME_KNOWLEDGE_JSON_PATH: ContextVar[Path | None] = ContextVar(
    "runtime_knowledge_json_path",
    default=None,
)
_RUNTIME_KNOWLEDGE_DATA_PATH: ContextVar[Path | None] = ContextVar(
    "runtime_knowledge_data_path",
    default=None,
)
_RUNTIME_EPISODE_NOTES: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "runtime_episode_notes",
    default=None,
)


def _effective_knowledge_paths() -> tuple[Path, Path]:
    json_path = _RUNTIME_KNOWLEDGE_JSON_PATH.get() or KNOWLEDGE_JSON_PATH
    txt_path = _RUNTIME_KNOWLEDGE_DATA_PATH.get() or KNOWLEDGE_DATA_PATH
    return json_path, txt_path


@contextmanager
def use_knowledge_paths(*, json_path: Path, txt_path: Path):
    token_json = _RUNTIME_KNOWLEDGE_JSON_PATH.set(Path(json_path))
    token_txt = _RUNTIME_KNOWLEDGE_DATA_PATH.set(Path(txt_path))
    try:
        yield
    finally:
        _RUNTIME_KNOWLEDGE_JSON_PATH.reset(token_json)
        _RUNTIME_KNOWLEDGE_DATA_PATH.reset(token_txt)


def _slugify(text: str, *, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return (slug or "fact")[:max_len]


def _normalize_entry(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(raw["id"]),
        "type": str(raw.get("type", "")).strip().upper(),
        "skill": str(raw.get("skill", "")).strip(),
        "recipe": str(raw.get("recipe", "")).strip(),
        "rules": str(raw.get("rules", "")).strip(),
    }


def _entry_key(entry: dict[str, Any]) -> tuple[str, str]:
    return (entry["type"], entry["skill"].lower())


def _is_durable_entry(entry: dict[str, Any]) -> bool:
    return entry.get("type") != "NOTE"


def _only_durable(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if _is_durable_entry(e)]


def render_knowledge_table(entries: list[dict[str, Any]]) -> str:
    if not entries:
        return _EMPTY_KNOWLEDGE_PLACEHOLDER

    headers = ("ID", "TYPE", "SKILL", "RECIPE", "RULES")
    rows: list[tuple[str, ...]] = []
    for entry in sorted(entries, key=lambda e: e["id"]):
        rows.append(
            (
                str(entry["id"]),
                entry["type"],
                entry["skill"],
                entry.get("recipe", ""),
                entry.get("rules", ""),
            )
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: tuple[str, ...]) -> str:
        parts = [f" {cells[i]:<{widths[i]}} " for i in range(len(cells))]
        return "|" + "|".join(parts) + "|"

    header_line = _fmt_row(headers)
    separator = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = "\n".join(_fmt_row(row) for row in rows)
    return "\n".join((header_line, separator, body))


def _load_persisted_knowledge_records() -> list[dict[str, Any]]:
    json_path, txt_path = _effective_knowledge_paths()
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if isinstance(entries, list):
            return [_normalize_entry(e) for e in entries if isinstance(e, dict) and "id" in e]

    if txt_path.exists():
        migrated = _parse_markdown_table(txt_path.read_text(encoding="utf-8"))
        if migrated:
            save_knowledge_records(_only_durable(migrated))
            return _only_durable(migrated)

    return []


def load_durable_knowledge_records() -> list[dict[str, Any]]:
    """Cross-episode knowledge persisted on disk (never includes TYPE=NOTE)."""
    entries = _load_persisted_knowledge_records()
    durable = _only_durable(entries)
    if len(durable) != len(entries):
        save_knowledge_records(durable)
    return durable


def load_episode_note_records() -> list[dict[str, Any]]:
    """Episode-only TYPE=NOTE rows (in-memory for the current run)."""
    notes = _RUNTIME_EPISODE_NOTES.get()
    if not notes:
        return []
    return [_normalize_entry(e) for e in notes]


def set_episode_note_records(notes: list[dict[str, Any]]) -> None:
    _RUNTIME_EPISODE_NOTES.set(
        [_normalize_entry(e) for e in notes if e.get("type") == "NOTE"]
    )


def clear_episode_notes() -> None:
    """Drop in-memory episode notes and remove any NOTE rows from persisted files."""
    set_episode_note_records([])
    durable = load_durable_knowledge_records()
    save_knowledge_records(durable)


def load_knowledge_records() -> list[dict[str, Any]]:
    """Durable knowledge plus episode notes (for prompts and merge during a run)."""
    return load_durable_knowledge_records() + load_episode_note_records()


def copy_durable_knowledge(
    *,
    source_json: Path,
    source_txt: Path,
    dest_json: Path,
    dest_txt: Path,
) -> None:
    """Copy persisted durable rows (TYPE != NOTE) from one knowledge file pair to another."""
    with use_knowledge_paths(json_path=source_json, txt_path=source_txt):
        entries = load_durable_knowledge_records()
    with use_knowledge_paths(json_path=dest_json, txt_path=dest_txt):
        save_knowledge_records(entries)


def read_starter_revision(json_path: Path) -> int:
    if not json_path.is_file():
        return 0
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        return int(payload.get("starter_revision") or 0)
    except (TypeError, ValueError):
        return 0


def write_starter_revision(json_path: Path, revision: int) -> None:
    if not json_path.is_file():
        return
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(payload, dict):
        return
    payload["starter_revision"] = int(revision)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_knowledge_records(entries: list[dict[str, Any]]) -> None:
    json_path, txt_path = _effective_knowledge_paths()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_entry(e) for e in entries]
    starter_revision = read_starter_revision(json_path)
    payload: dict[str, Any] = {"version": 1, "entries": normalized}
    if starter_revision:
        payload["starter_revision"] = starter_revision
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    rendered = render_knowledge_table(normalized)
    txt_path.write_text(
        rendered + ("\n" if rendered else ""),
        encoding="utf-8",
    )


def load_knowledge() -> str:
    _, txt_path = _effective_knowledge_paths()
    entries = load_knowledge_records()
    if not entries:
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            if text and text != _EMPTY_KNOWLEDGE_PLACEHOLDER:
                return text
        return _EMPTY_KNOWLEDGE_PLACEHOLDER
    return render_knowledge_table(entries)


def save_knowledge(body: str) -> None:
    """Backward-compatible: parse rendered table or legacy lines into JSON."""
    entries = _parse_markdown_table(body)
    if not entries:
        entries = _parse_legacy_lines(body)
    save_knowledge_records(_only_durable(entries))


def _parse_markdown_table(text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or _TABLE_SEPARATOR_RE.match(stripped):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 5:
            continue
        if cells[0].upper() in {"ID", ""}:
            continue
        try:
            row_id = int(cells[0])
        except ValueError:
            continue
        entries.append(
            _normalize_entry(
                {
                    "id": row_id,
                    "type": cells[1],
                    "skill": cells[2],
                    "recipe": cells[3],
                    "rules": cells[4],
                }
            )
        )
    return entries


def _parse_legacy_lines(text: str) -> list[dict[str, Any]]:
    if text.strip() == _EMPTY_KNOWLEDGE_PLACEHOLDER:
        return []

    entries: list[dict[str, Any]] = []
    next_id = 1
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        prefix, remainder = stripped.split(":", 1)
        prefix = prefix.strip().upper()
        remainder = remainder.strip()
        if prefix not in _LEGACY_PREFIXES or not remainder:
            continue
        if prefix in {"CORRECTION", "UPDATE"}:
            if ":" in remainder:
                inner_prefix, inner_rest = remainder.split(":", 1)
                prefix = inner_prefix.strip().upper()
                remainder = inner_rest.strip()
            else:
                continue
        if prefix not in _KNOWLEDGE_TYPES:
            continue
        skill = _slugify(remainder)
        entry = _normalize_entry(
            {
                "id": next_id,
                "type": prefix,
                "skill": skill,
                "recipe": remainder if prefix == "RECIPE" else "",
                "rules": remainder if prefix != "RECIPE" else "",
            }
        )
        entries.append(entry)
        next_id += 1
    return entries


def _iter_field_records(block: str):
    """Yield one field dict per UPSERT/DELETE record in a <to_database> block."""
    current: list[str] = []

    def _flush() -> dict[str, str] | None:
        if not current:
            return None
        fields = _parse_field_block("\n".join(current))
        current.clear()
        return fields or None

    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "---":
            fields = _flush()
            if fields:
                yield fields
            continue
        upper = stripped.upper()
        if current and (upper.startswith("TYPE=") or upper.startswith("OP=")):
            fields = _flush()
            if fields:
                yield fields
        current.append(stripped)

    fields = _flush()
    if fields:
        yield fields


def _parse_field_block(block: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("|"):
            continue
        if "=" in stripped:
            key, value = stripped.split("=", 1)
            fields[key.strip().upper()] = value.strip()
            continue
        if ":" in stripped:
            prefix, remainder = stripped.split(":", 1)
            prefix = prefix.strip().upper()
            remainder = remainder.strip()
            if prefix in _LEGACY_PREFIXES and remainder:
                fields["TYPE"] = prefix if prefix in _KNOWLEDGE_TYPES else ""
                if prefix in {"CORRECTION", "UPDATE"} and ":" in remainder:
                    inner_prefix, inner_rest = remainder.split(":", 1)
                    fields["TYPE"] = inner_prefix.strip().upper()
                    remainder = inner_rest.strip()
                if fields.get("TYPE") == "RECIPE":
                    fields["RECIPE"] = remainder
                    fields.setdefault("SKILL", _slugify(remainder))
                else:
                    fields["RULES"] = remainder
                    fields.setdefault("SKILL", _slugify(remainder))
    return fields


def merge_knowledge_entries(
    existing: list[dict[str, Any]],
    new_block: str,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {
        _entry_key(e): dict(e) for e in existing
    }
    next_id = max((e["id"] for e in existing), default=0) + 1

    def _upsert(fields: dict[str, str]) -> None:
        nonlocal next_id
        op = fields.get("OP", "UPSERT").upper()
        type_ = fields.get("TYPE", "").upper()
        skill = fields.get("SKILL", "").strip()
        if op == "DELETE":
            if type_ and skill:
                by_key.pop((type_, skill.lower()), None)
            return
        if not type_ or not skill:
            return
        if type_ not in _KNOWLEDGE_TYPES:
            return

        key = (type_, skill.lower())
        recipe = fields.get("RECIPE", "")
        rules = fields.get("RULES", "")
        if key in by_key:
            entry = dict(by_key[key])
            if recipe:
                entry["recipe"] = recipe
            if rules:
                entry["rules"] = rules
        else:
            entry = _normalize_entry(
                {
                    "id": next_id,
                    "type": type_,
                    "skill": skill,
                    "recipe": recipe,
                    "rules": rules,
                }
            )
            next_id += 1
        by_key[key] = entry

    block = new_block.strip()
    if not block:
        return sorted(by_key.values(), key=lambda e: e["id"])

    if block.startswith("{"):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if "entries" in payload and isinstance(payload["entries"], list):
                for item in payload["entries"]:
                    if isinstance(item, dict):
                        _upsert(
                            {
                                k.upper(): str(v)
                                for k, v in item.items()
                                if k.upper() != "ID"
                            }
                        )
                return sorted(by_key.values(), key=lambda e: e["id"])
            _upsert({k.upper(): str(v) for k, v in payload.items() if k != "id"})

    records = list(_iter_field_records(block))
    if records:
        for fields in records:
            if fields.get("TYPE") and not fields.get("SKILL"):
                remainder = fields.get("RULES") or fields.get("RECIPE", "")
                if remainder:
                    fields["SKILL"] = _slugify(remainder)
            if fields.get("TYPE"):
                _upsert(fields)
    else:
        for line in block.splitlines():
            line_fields = _parse_field_block(line)
            if line_fields.get("TYPE"):
                if not line_fields.get("SKILL"):
                    remainder = line_fields.get("RULES") or line_fields.get("RECIPE", "")
                    if remainder:
                        line_fields["SKILL"] = _slugify(remainder)
                _upsert(line_fields)

    return sorted(by_key.values(), key=lambda e: e["id"])


def merge_knowledge(existing: str, new_block: str) -> str:
    """
    Merge durable knowledge (legacy string API).

    Accepts markdown tables, key=value blocks, or legacy prefixed lines.
    Returns a rendered markdown table.
    """
    if existing.strip() == _EMPTY_KNOWLEDGE_PLACEHOLDER:
        existing = ""

    current = _parse_markdown_table(existing)
    if not current:
        current = _parse_legacy_lines(existing)

    merged = merge_knowledge_entries(current, new_block)
    if not merged:
        return ""
    return render_knowledge_table(merged)


def extract_to_database_blocks(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(1).strip() for m in _TO_DATABASE_PATTERN.finditer(text) if m.group(1).strip()]


def strip_to_database_tags(text: str) -> str:
    if not text:
        return text
    cleaned = _TO_DATABASE_PATTERN.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def snapshot_reasoning_base(force: bool = False) -> Path | None:
    """
    Copy reasoning.txt to reasoning_base_YYYYMMDD_HHMMSS.txt when the template
    is newer than the latest snapshot (or when force=True).
    """
    if not REASONING_TEMPLATE_PATH.exists():
        return None

    DATABASE_FORMULATION_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(DATABASE_FORMULATION_DIR.glob("reasoning_base_*.txt"))
    template_mtime = REASONING_TEMPLATE_PATH.stat().st_mtime

    if existing and not force:
        latest = existing[-1]
        if latest.stat().st_mtime >= template_mtime:
            return latest

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = DATABASE_FORMULATION_DIR / f"reasoning_base_{stamp}.txt"
    dest.write_text(REASONING_TEMPLATE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def apply_knowledge_from_response(raw_response: str) -> Tuple[str, bool, str]:
    """
    Extract <to_database> blocks and merge into knowledge stores.

    Durable types are written to knowledge_data.json/.txt; TYPE=NOTE stays in
    episode memory only (see clear_episode_notes at run start).

    Returns (cleaned_response, knowledge_updated, combined_new_block_text).
    """
    blocks = extract_to_database_blocks(raw_response)
    cleaned = strip_to_database_tags(raw_response)
    if not blocks:
        return cleaned, False, ""

    combined = "\n".join(blocks)
    current_durable = load_durable_knowledge_records()
    current_notes = load_episode_note_records()
    current_entries = current_durable + current_notes
    merged_entries = merge_knowledge_entries(current_entries, combined)
    if not merged_entries and not current_entries:
        return cleaned, False, combined

    new_durable = _only_durable(merged_entries)
    new_notes = [e for e in merged_entries if e.get("type") == "NOTE"]
    durable_changed = new_durable != current_durable
    notes_changed = new_notes != current_notes

    if not durable_changed and not notes_changed:
        return cleaned, False, combined

    if durable_changed:
        save_knowledge_records(new_durable)
        snapshot_reasoning_base()
    if notes_changed:
        set_episode_note_records(new_notes)
    return cleaned, True, combined


def process_agent_knowledge(raw_response: str) -> Tuple[str, bool]:
    """Backward-compatible wrapper around apply_knowledge_from_response."""
    cleaned, updated, _ = apply_knowledge_from_response(raw_response)
    return cleaned, updated
