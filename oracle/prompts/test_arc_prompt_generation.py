from oracle.prompts.prompt_generation import (
    GAME_KIND_ARC_AGI,
    coerce_megaprompt_config_for_world_mode,
    generate_arc_agent_prompt,
    list_megaprompt_configs_for_world_mode,
)


def _arc_obs():
    return {
        "game_id": "ls20",
        "title": "LS20",
        "state": "NOT_FINISHED",
        "levels_completed": 0,
        "available_actions": ["ACTION1", "ACTION6"],
        "frame_grid": "01\nef",
        "png_b64": "QUJD",
        "w": 64,
        "h": 64,
    }


def _arc_obs_with_previous():
    obs = dict(_arc_obs())
    obs.update(
        {
            "previous_png_b64": "UFJFVg==",
            "previous_w": 64,
            "previous_h": 64,
            "previous_action": "ACTION1",
        }
    )
    return obs


def test_arc_megaprompt_options_are_separate_family():
    assert list_megaprompt_configs_for_world_mode("craftax", game_kind=GAME_KIND_ARC_AGI) == [
        "arc_2_image",
        "arc_grid",
        "arc_grid_image",
        "arc_image",
    ]
    assert (
        coerce_megaprompt_config_for_world_mode(
            "database_formulation",
            "craftax",
            game_kind=GAME_KIND_ARC_AGI,
        )
        == "arc_grid"
    )


def test_arc_grid_prompt_contains_text_frame_grid():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        current_step=7,
        megaprompt_config_name="arc_grid",
    )
    assert "## Current step\n7" in prompt
    assert "Current frame grid:" in prompt
    assert "01\nef" in prompt
    assert "[[image:" not in prompt


def test_arc_prompt_describes_controller_actions_and_click_format():
    obs = dict(_arc_obs())
    obs["available_actions"] = [
        "ACTION1",
        "ACTION2",
        "ACTION3",
        "ACTION4",
        "ACTION5",
        "ACTION6",
        "ACTION7",
    ]
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=obs,
        megaprompt_config_name="arc_grid",
    )
    assert "ACTION1: Up arrow on the game controller." in prompt
    assert "ACTION2: Down arrow on the game controller." in prompt
    assert "ACTION3: Left arrow on the game controller." in prompt
    assert "ACTION4: Right arrow on the game controller." in prompt
    assert "ACTION6 x y: click on the game frame" in prompt
    assert "Use exactly `ACTION6 32 31` or `ACTION6 [32,31]`" in prompt
    assert "Use the ACTION names, not words like LEFT or CLICK." in prompt


def test_arc_image_prompt_contains_image_marker():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        megaprompt_config_name="arc_image",
    )
    assert "Frame image: 64x64 PNG" in prompt
    assert "[[image:data:image/png;base64,QUJD]]" in prompt
    assert "Current frame grid:" not in prompt


def test_arc_grid_image_prompt_contains_grid_and_image_marker():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        megaprompt_config_name="arc_grid_image",
    )
    assert "Current frame grid:" in prompt
    assert "01\nef" in prompt
    assert "Frame image: 64x64 PNG" in prompt
    assert "[[image:data:image/png;base64,QUJD]]" in prompt
    assert "Use the image for spatial visual layout" in prompt


def test_arc_2_image_prompt_contains_previous_and_current_images():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs_with_previous(),
        megaprompt_config_name="arc_2_image",
    )
    assert "Latest environment action before the current frame: ACTION1." in prompt
    assert "Previous observation image: 64x64 PNG" in prompt
    assert "Current observation image: 64x64 PNG" in prompt
    assert "[[image:data:image/png;base64,UFJFVg==]]" in prompt
    assert "[[image:data:image/png;base64,QUJD]]" in prompt
    assert prompt.count("[[image:") == 2
    assert "compare them to infer what the latest environment" in prompt


def test_arc_prompt_does_not_open_reasoning_block():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        megaprompt_config_name="arc_grid",
    )
    assert not prompt.rstrip().endswith("--- REASONING ---")
    assert "If you include reasoning, put it before the final block." in prompt


def test_arc_prompt_contains_full_operator_dialog():
    dialog = [
        {"question": f"Question {i}", "answer": f"Answer {i}"}
        for i in range(1, 5)
    ]
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        dialog=dialog,
        megaprompt_config_name="arc_grid",
    )
    assert "Turn 1 Agent: Question 1" in prompt
    assert "Turn 1 Human operator: Answer 1" in prompt
    assert "Turn 4 Agent: Question 4" in prompt
    assert "Turn 4 Human operator: Answer 4" in prompt
    assert "Do not ask a question that" in prompt


def test_arc_prompt_preserves_operator_dialog_step_numbers():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        dialog=[
            {
                "question": "What does ACTION1 do?",
                "answer": "It moves up.",
                "question_step": 3,
                "answer_step": 3,
            }
        ],
        megaprompt_config_name="arc_grid",
    )
    assert "Turn 1 Agent (asked at step 3): What does ACTION1 do?" in prompt
    assert "Turn 1 Human operator (answered at step 3): It moves up." in prompt


def test_arc_prompt_encourages_operator_questions_for_hidden_rules():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        megaprompt_config_name="arc_grid",
    )
    assert "Use the human operator to discover them." in prompt
    assert "what an action does" in prompt
    assert "what objective" in prompt
    assert "what you are missing" in prompt
    assert "using as few environment actions as possible" in prompt
    assert "strongly prefer asking the operator" in prompt


def test_arc_prompt_contains_last_thirty_actions():
    actions = [f"ACTION{i}" for i in range(1, 36)]
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        action_history=actions,
        megaprompt_config_name="arc_grid",
    )
    assert "## Recent action history" in prompt
    assert "1. ACTION6" in prompt
    assert "30. ACTION35" in prompt
    assert "ACTION5" not in prompt


def test_arc_prompt_does_not_warn_before_four_repeated_actions():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        action_history=["ACTION1", "ACTION1", "ACTION1"],
        megaprompt_config_name="arc_grid",
    )
    assert "## Repeated action tip" in prompt
    assert "TIP!!!!" not in prompt
    assert "No repeated action pattern detected." in prompt


def test_arc_prompt_warns_after_four_repeated_actions():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        action_history=["ACTION2", "ACTION1", "ACTION1", "ACTION1", "ACTION1"],
        megaprompt_config_name="arc_grid",
    )
    assert "TIP!!!! You have executed ACTION1 for 4 consecutive environment actions." in prompt
    assert "This may mean you are stuck, not moving, or not making progress." in prompt
    assert "Strongly consider asking the human operator" in prompt
    assert "instead of repeating the same action again" in prompt


def test_arc_prompt_contains_previous_agent_reasoning():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        previous_reasoning="I tried ACTION1 and the blue shape moved.",
        megaprompt_config_name="arc_grid",
    )
    assert "## Previous agent reasoning" in prompt
    assert "I tried ACTION1 and the blue shape moved." in prompt


def test_arc_image_prompt_contains_previous_agent_reasoning():
    prompt = generate_arc_agent_prompt(
        goal="complete the level",
        arc_observation=_arc_obs(),
        previous_reasoning="The door shape did not match yet.",
        megaprompt_config_name="arc_image",
    )
    assert "## Previous agent reasoning" in prompt
    assert "The door shape did not match yet." in prompt
