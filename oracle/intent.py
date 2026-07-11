import logging
import os
import re
from typing import Any, List, Optional, Sequence

import numpy as np

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

logger = logging.getLogger(__name__)

# Matches a tile like [12, 3] or (12, 3); same idea as PathExpertPipeline._extract_target_location_from_text.
_BRACKET_COORD_RE = re.compile(r"\[\s*-?\d+\s*,\s*-?\d+\s*\]")


def _question_has_bracket_coordinates(text: str) -> bool:
    return bool(text and _BRACKET_COORD_RE.search(text))


def _looks_like_path_navigation_question(text: str) -> bool:
    """
    Path pipeline needs a concrete target tile (parsed from question, env, or helper).
    For routing we only treat as path_expert when the question itself names coordinates
    and reads like navigation (see PathExpertPipeline.answer).
    """
    if not _question_has_bracket_coordinates(text):
        return False
    lower = text.lower()
    stripped = lower.lstrip()
    if stripped.startswith("where is ") or stripped.startswith("where are "):
        return False
    hints = (
        "navigate",
        "direction",
        "reach the block",
        "reach the target",
        "reach the point",
        "go to ",
        "goal:",
        "obstacle",
        "show path",
        "closer to",
        "toward the target",
        "coordinates",
        "get to ",
        "move to ",
        "path to ",
        "route to ",
        "get closer",
        "landmark",
        "right path",
        "on the right track",
        "waypoint",
    )
    return any(h in lower for h in hints)


# Training examples used by train_intent.py and by __main__ below.
# Single list of (text, label) pairs so texts and labels cannot drift apart.
#
# Expert capabilities (keep labels aligned with oracle behavior):
# - map_expert: global map / where objects or tiles are (not turn-by-turn navigation).
# - path_expert: route to a fixed tile; pipeline parses an explicit [row, col] (or x/y)
#   from the question — a leading "goal: go to the point ..." line is optional, not required.
# - action_expert: only actions listed in oracle/prompts/texts/action_prompt.txt /
#   ALLOWED_AGENT_ACTIONS in prompt_generation.py (e.g. MAKE_WOOD_PICKAXE, not CRAFT_*).
INTENT_TRAIN_EXAMPLES: list[tuple[str, str]] = [
    # --- map_expert: global positions / layout (where on the map, not how to walk there) ---
    ("Where is the nearest coal?", "map_expert"),
    ("Where is the nearest tree?", "map_expert"),
    ("Where is the nearest crafting table?", "map_expert"),
    ("Where is the nearest water?", "map_expert"),
    ("Where is iron ore on the map relative to the agent?", "map_expert"),
    ("What block is nearest to the agent?", "map_expert"),
    ("What block is nearest to the agent except grass?", "map_expert"),
    ("What blocks are around the agent on the global map?", "map_expert"),
    ("Where is located target point?", "map_expert"),
    ("Where is the target point?", "map_expert"),
    # --- path_expert: only when the question names a destination tile [row, col] ---
    (
        "goal: go to the point [20, 2]. In which direction should I go now?",
        "path_expert",
    ),
    (
        "goal: go to the point [20, 3]. In which direction should I go now?",
        "path_expert",
    ),
    (
        "goal: go to the point [19, 3]. How do I move around this obstacle toward the target?",
        "path_expert",
    ),
    (
        "goal: go to the point [2, 34]. What should I do next to reach the target?",
        "path_expert",
    ),
    (
        "I need to reach tile [19, 3]. How do I move around an obstacle toward that point?",
        "path_expert",
    ),
    ("How can I navigate to coordinates [45, 54]?", "path_expert"),
    (
        "In which direction should I move to reach the block at [23, 45]?",
        "path_expert",
    ),
    ("How can I reach the point with coordinates [23, 34]?", "path_expert"),
    ("What should I do next to get closer to [10, 15]?", "path_expert"),
    ("Show path from my position to the goal cell [12, 40].", "path_expert"),
    (
        "What landmark should I use to know I am on the right path to [23, 45]?",
        "path_expert",
    ),
    # --- mechanics_expert: recipes, progression, what to gather or craft (not env navigation) ---
    ("What achievements are required before collecting coal?", "mechanics_expert"),
    (
        "What achievements are required before making an iron sword?",
        "mechanics_expert",
    ),
    ("How to make a stone sword?", "mechanics_expert"),
    ("How to make an iron sword?", "mechanics_expert"),
    ("How to make a diamond sword?", "mechanics_expert"),
    ("What do I need to gather coal?", "mechanics_expert"),
    ("Which instruments do I need to make a pickaxe?", "mechanics_expert"),
    ("How can I make a crafting table?", "mechanics_expert"),
    ("How do I craft a wooden pickaxe?", "mechanics_expert"),
    ("How can I find stone deposits to mine?", "mechanics_expert"),
    ("How do I progress from wood tools to stone tools?", "mechanics_expert"),
    # --- action_expert: preconditions for a single allowed action token ---
    ("What action do i need to use to collect wood?", "action_expert"),
    ("What action do i need to use to collect stone?", "action_expert"),
    ("What action do i need to use to place table?", "action_expert"),
    ("What is needed to PLACE_PLANT?", "action_expert"),
    ("What is required to DO (TO GATHER SOMETHING)?", "action_expert"),
    ("What is required to DO (TO FIGHT)?", "action_expert"),
    ("What conditions are required to DO (DRINK WATER)?", "action_expert"),
    ("What I need to implement action DO (TO GATHER SOMETHING)?", "action_expert"),
    ("What do I need to perform PLACE_TABLE?", "action_expert"),
    ("What do I need to perform MAKE_WOOD_PICKAXE?", "action_expert"),
    ("What is needed to perform MAKE_STONE_SWORD?", "action_expert"),
    ("What is required to perform UP?", "action_expert"),
    ("What is needed to perform REST?", "action_expert"),
    # --- question / goal experts (meta planning) ---
    ("Goal: Collect coal. What questions to ask the experts?", "question_expert"),
    ("Goal: Collect diamond. What questions to ask the experts?", "question_expert"),
    ("Goal: Make a crafting table. What questions to ask the experts", "question_expert"),
    ("Goal: Collect coal. How to achieve the goal?", "goal_expert"),
    ("Goal: Collect diamond. How to achieve the goal?", "goal_expert"),
    ("Goal: Make a crafting table. How to achieve the goal?", "goal_expert"),
]

INTENT_TRAIN_TEXTS = [text for text, _ in INTENT_TRAIN_EXAMPLES]
INTENT_TRAIN_LABELS = [label for _, label in INTENT_TRAIN_EXAMPLES]
assert len(INTENT_TRAIN_TEXTS) == len(INTENT_TRAIN_LABELS)
assert len(INTENT_TRAIN_TEXTS) == len(INTENT_TRAIN_EXAMPLES)


class Intent:
    MAP_EXPERT = "map_expert"
    MECHANICS_EXPERT = "mechanics_expert"
    ACTION_EXPERT = "action_expert"
    QUESTION_EXPERT = "question_expert"
    GOAL_EXPERT = "goal_expert"
    PATH_EXPERT = "path_expert"
    ALL_EXPERTS = (
        MAP_EXPERT,
        MECHANICS_EXPERT,
        ACTION_EXPERT,
        QUESTION_EXPERT,
        GOAL_EXPERT,
        PATH_EXPERT,
    )
    _ALIAS_TO_EXPERT = {
        "map": MAP_EXPERT,
        "mechanics": MECHANICS_EXPERT,
        "action": ACTION_EXPERT,
        "question": QUESTION_EXPERT,
        "goal": GOAL_EXPERT,
        "path": PATH_EXPERT,
    }

    def __init__(self, model_name: Optional[str] = None, *, use_model: bool = False):
        self.model: Any = None
        self.model_name = model_name
        self._setfit_import_error: Optional[Exception] = None
        if use_model and model_name:
            self._load_model()

    def _load_model(self) -> None:
        if not self.model_name:
            return
        try:
            from setfit import SetFitModel  # pyright: ignore[reportMissingImports]

            self.model = SetFitModel.from_pretrained(self.model_name)
        except Exception as exc:
            self._setfit_import_error = exc
            self.model = None
            logger.warning(
                "SetFit intent model is unavailable; using heuristic intent routing. "
                "Original error: %s",
                exc,
            )

    def train(self, 
              train_texts: list[str],
              train_labels: list[str],
              num_iterations: int = 20,
              batch_size: int = 16,
              save_path: str = "./intent_model/oracle_intent_model"):
        if self.model is None:
            raise RuntimeError(
                "Intent model is not available. Install compatible setfit/transformers "
                "packages before training."
            ) from self._setfit_import_error
        from datasets import Dataset
        from setfit import SetFitTrainer  # pyright: ignore[reportMissingImports]
        
        train_ds = Dataset.from_dict({"text": train_texts, 
                                      "label": train_labels})
        self.train_dataset = train_ds
        
        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            num_iterations=num_iterations,
            batch_size=batch_size,
        )
        trainer.train()
        trainer.model.save_pretrained(save_path)
        return self.model

    def _get_model_classes(self) -> List[str]:
        classes = []
        model_head = getattr(self.model, "model_head", None)
        if model_head is not None and hasattr(model_head, "classes_"):
            classes = [str(x) for x in list(model_head.classes_)]
        if not classes and hasattr(self.model, "labels"):
            labels = getattr(self.model, "labels")
            classes = [str(x) for x in labels] if labels is not None else []
        return classes

    @classmethod
    def normalize_expert_name(cls, expert: str) -> str:
        normalized = str(expert).strip().lower()
        if normalized in cls._ALIAS_TO_EXPERT:
            return cls._ALIAS_TO_EXPERT[normalized]
        if normalized in cls.ALL_EXPERTS:
            return normalized
        raise ValueError(
            f"Unknown expert '{expert}'. "
            f"Available experts: {', '.join(cls.ALL_EXPERTS)}"
        )

    @classmethod
    def normalize_allowed_experts(cls, allowed_experts: Sequence[str]) -> List[str]:
        normalized: List[str] = []
        for expert in allowed_experts:
            expert_name = cls.normalize_expert_name(expert)
            if expert_name not in normalized:
                normalized.append(expert_name)
        if not normalized:
            raise ValueError("allowed_experts cannot be empty")
        return normalized

    def predict(
        self,
        text: str,
        allowed_experts: Optional[Sequence[str]] = None,
    ):
        if self.model is None:
            return np.array([self._predict_heuristic(text, allowed_experts=allowed_experts)])
        if not allowed_experts:
            return self.model.predict([text])

        normalized_allowed = self.normalize_allowed_experts(allowed_experts)
        if len(normalized_allowed) == 1:
            return np.array([normalized_allowed[0]])

        predicted_proba = self.model.predict_proba([text])
        probs = np.asarray(predicted_proba)[0]
        classes = self._get_model_classes()

        if not classes or len(classes) != len(probs):
            # Fallback when class names are unavailable: choose unrestricted top-1.
            return self.model.predict([text])

        class_to_prob = {label: float(prob) for label, prob in zip(classes, probs)}
        available_allowed = [label for label in normalized_allowed if label in class_to_prob]
        if not available_allowed:
            raise ValueError(
                "None of allowed_experts are present in intent model classes. "
                f"Allowed: {normalized_allowed}; model classes: {classes}"
            )

        best_label = max(available_allowed, key=lambda label: class_to_prob[label])
        return np.array([best_label])

    @classmethod
    def _predict_heuristic(
        cls,
        text: str,
        allowed_experts: Optional[Sequence[str]] = None,
    ) -> str:
        normalized_allowed = (
            cls.normalize_allowed_experts(allowed_experts)
            if allowed_experts
            else list(cls.ALL_EXPERTS)
        )
        lower = str(text or "").lower()
        raw = str(text or "")

        if cls.PATH_EXPERT in normalized_allowed and _looks_like_path_navigation_question(raw):
            return cls.PATH_EXPERT

        # "How / where to find <resource>" without a coordinate goal → mechanics, not path.
        if (
            cls.MECHANICS_EXPERT in normalized_allowed
            and not _question_has_bracket_coordinates(raw)
            and "find" in lower
            and any(
                w in lower
                for w in (
                    "stone",
                    "coal",
                    "iron",
                    "ore",
                    "diamond",
                    "wood",
                    "water",
                    "lava",
                    "deposit",
                )
            )
        ):
            return cls.MECHANICS_EXPERT

        ordered_rules = [
            (
                cls.MAP_EXPERT,
                (
                    "nearest",
                    "where is",
                    "where are",
                    "on the map",
                    "global map",
                    "render map",
                    "what block",
                    "blocks are around",
                    "around agent",
                    "located",
                    "relative to the agent",
                ),
            ),
            (
                cls.ACTION_EXPERT,
                (
                    "required to ",
                    "required for",
                    "conditions are",
                    "what conditions",
                    "what is needed to ",
                    "what is needed to perform",
                    "what do i need to perform",
                    "what is required to perform",
                    "implement action",
                    "prerequisite",
                ),
            ),
            (
                cls.MECHANICS_EXPERT,
                (
                    "craft",
                    "recipe",
                    "achievement",
                    "how to make",
                    "gather",
                    "instruments",
                    "pickaxe",
                    "crafting table",
                    "progress from",
                    "deposits to mine",
                ),
            ),
            (cls.QUESTION_EXPERT, ("what questions", "which questions", "ask the experts")),
            (cls.GOAL_EXPERT, ("how to achieve the goal", "strategy", "plan")),
        ]
        for expert, keywords in ordered_rules:
            if expert not in normalized_allowed:
                continue
            if any(keyword in lower for keyword in keywords):
                return expert
        # Steering question with no destination tile → not path; prefer mechanics over a random default.
        if (
            cls.MECHANICS_EXPERT in normalized_allowed
            and "in which direction" in lower
            and not _question_has_bracket_coordinates(raw)
        ):
            return cls.MECHANICS_EXPERT
        return normalized_allowed[0]
    
    

if __name__ == "__main__":
    intent = Intent(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    intent.train(train_texts=INTENT_TRAIN_TEXTS,
                 train_labels=INTENT_TRAIN_LABELS,
                 num_iterations=20,
                 batch_size=16,
    
                 save_path="./intent_model/oracle_intent_model")
    
    print(intent.predict("Where is the nearest diamond?"))
    print(intent.predict("How to make crafting table?"))
    
    intent = Intent(model_name="./intent_model/oracle_intent_model")
    print(intent.predict("Where is the nearest diamond?"))
    print(intent.predict("How to make crafting table?"))
    
    print(intent.predict("goal: go to the point [20, 3]. In which direction should I go now?"))