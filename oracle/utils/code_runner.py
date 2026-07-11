import importlib
import sys
import textwrap
from pathlib import Path


def _save_code(code: str, module_name: str) -> None:
    path = Path("./answer")
    path.mkdir(exist_ok=True)
    with open(path / f"{module_name}.py", "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(code))


def _import_answer_module(module_name: str):
    if "./answer" not in sys.path:
        sys.path.append("./answer")
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return module


def run_llm_code(code: str, state, module_name: str = "answer_code"):
    _save_code(code, module_name)
    module = _import_answer_module(module_name)
    return module.checker(state)


def run_llm_mechanics_code(code: str, module_name: str = "answer_code"):
    _save_code(code, module_name)
    module = _import_answer_module(module_name)
    return module.checker()
