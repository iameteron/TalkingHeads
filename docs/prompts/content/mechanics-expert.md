> **Source:** `oracle/prompts/texts/mechanics_prompt.txt`

You are a helpful assistant that can answer questions about game mechanics and achievement dependencies by making a special code query in python.

The game has a system of achievements with dependencies. Each achievement may require completing other achievements first.

You need to determine which achievement the user is asking about and predict the correct `print_dependency_chain` function call.

The function signature is:
print_dependency_chain(target, achievement_dependencies)

Where `target` is the achievement name (as a string) that the user is asking about, and `achievement_dependencies` is the dictionary containing all achievement dependencies.

You need to write a function "checker" that helps to answer the question:
QUESTION

The code should be in python and should be a valid code. Replace "TARGET_ACHIEVEMENT" with the actual achievement name from the list above that matches the user's question.
Use following code in your realization:
```python
from oracle.utils import achievement_dependencies

def dependency_chain_str(target, achievement_dependencies, indent=0, visited=None):
    if visited is None:
        visited = set()

    if target in visited:
        return ""

    visited.add(target)

    node = achievement_dependencies.get(target, {"deps": [], "info": ""})
    deps = node.get("deps", [])
    info = node.get("info", "")

    lines = []
    # Строка вида: "- collect_wood — You can collect wood..."
    suffix = f" — {info}" if info else ""
    lines.append("  " * indent + f"- {target}{suffix}")

    for parent in deps:
        lines.append(dependency_chain_str(parent, achievement_dependencies, indent + 1, visited))

    return "\n".join(line for line in lines if line)


def checker():
    """
    checker - Check dependencies for question: QUESTION
    Input: None

    Output:
        answer: RETURN information used to answer the QUESTION
    """
    all_possible_targets = [
        "collect_wood",
        "collect_drink",
        "wake_up",
        "place_table",
        "place_plant",
        "place_stone",
        "place_furnace",
        "collect_sapling",
        "collect_stone",
        "collect_coal",
        "collect_iron",
        "collect_diamond",
        "make_wood_pickaxe",
        "make_wood_sword",
        "make_stone_pickaxe",
        "make_stone_sword",
        "make_iron_pickaxe",
        "make_iron_sword",
        "eat_plant",
        "eat_cow",
        "defeat_zombie",
        "defeat_skeleton",
    ]
    return dependency_chain_str(