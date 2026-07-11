"""
Benchmarking script for Oracle - Active agent interactions performance.

This script tests specific instructions multiple times and measures Success Rate (SR).
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import json
import jax
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for logging
import matplotlib.pyplot as plt

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.craftax_state import EnvState
from craftax.craftax.constants import Action

from oracle.prompts.prompt_generation import (
    generate_agent_prompt,
    build_previous_actions_analysis,
)
from oracle.active_agent_base import ActiveAgent
from oracle.configs import GenConfig
from oracle.config_loader import load_config
from oracle.knowledge import clear_episode_notes
from oracle.oracle import Oracle
from oracle.statistics_wrapper import ActiveAgentStatistics

from benchmarking_utils import (
    check_instruction_completed,
    INSTRUCTION_CHECKERS,
    parse_agent_answer,
    format_inventory_from_state,
    format_observation_from_state,
)

# Optional Comet ML import

from comet_ml import Experiment
COMET_AVAILABLE = True


# Instructions to benchmark
INSTRUCTIONS = [
  #   "Collect wood",
   #  "Place table",
  #  "Make wooden pickaxe",
    "Dig a stone from rock",
    "Place stone",
]

# Number of runs per instruction
NUM_RUNS_PER_INSTRUCTION = 10

# Maximum steps per run
MAX_STEPS_PER_RUN = 130


def _tile_name_from_id(block_id: int) -> str:
    """Convert numeric block id to a human‑readable name like 'Stone', 'Tree', 'Path'."""
    from craftax.craftax.constants import BlockType
    try:
        bt = BlockType(block_id)
    except ValueError:
        return "Unknown"
    return bt.name.title().replace("_", " ")


def _iter_non_grass_tiles(state: EnvState) -> List[Tuple[str, Tuple[int, int]]]:
    """
    Return a list of (tile_name, (x, y)) for all tiles that are NOT
    out-of-bounds / darkness / grass.
    """
    from craftax.craftax.constants import BlockType
    
    # Convert map to numpy array once to avoid repeated JAX operations
    # This is more memory-efficient than accessing individual elements
    map_array = np.array(state.map)
    h, w = map_array.shape
    tiles: List[Tuple[str, Tuple[int, int]]] = []

    for x in range(h):
        for y in range(w):
            block_id = int(map_array[x, y])
            if block_id in (
                BlockType.OUT_OF_BOUNDS.value,
                BlockType.DARKNESS.value,
                BlockType.GRASS.value,
            ):
                continue
            name = _tile_name_from_id(block_id)
            tiles.append((name, (x, y)))

    return tiles


def format_inventory_from_state(state: EnvState) -> str:
    """Build text description of the agent's inventory (items with count > 0)."""
    inv = state.inventory
    items: List[str] = []
    if inv.wood > 0:
        items.append(f"wood: {int(inv.wood)}")
    if inv.stone > 0:
        items.append(f"stone: {int(inv.stone)}")
    if inv.coal > 0:
        items.append(f"coal: {int(inv.coal)}")
    if inv.iron > 0:
        items.append(f"iron: {int(inv.iron)}")
    if inv.diamond > 0:
        items.append(f"diamond: {int(inv.diamond)}")
    if inv.sapling > 0:
        items.append(f"sapling: {int(inv.sapling)}")
    if inv.wood_pickaxe > 0:
        items.append(f"wood_pickaxe: {int(inv.wood_pickaxe)}")
    if inv.stone_pickaxe > 0:
        items.append(f"stone_pickaxe: {int(inv.stone_pickaxe)}")
    if inv.iron_pickaxe > 0:
        items.append(f"iron_pickaxe: {int(inv.iron_pickaxe)}")
    if inv.wood_sword > 0:
        items.append(f"wood_sword: {int(inv.wood_sword)}")
    if inv.stone_sword > 0:
        items.append(f"stone_sword: {int(inv.stone_sword)}")
    if inv.iron_sword > 0:
        items.append(f"iron_sword: {int(inv.iron_sword)}")
    return ", ".join(items) if items else "Empty"


def format_observation_from_state(state: EnvState, k: int = 5) -> str:
    """
    Build a short text observation like:

        You in [32, 32]
        Stone in [30, 31]
        Tree in [29, 28]
        Path in [28, 28]
        ...

        The rest is grass. (You can walk only on grass or path)

    Shows up to k unique nearest object types (nearest instance of each type).
    """
    # Player position (world coordinates)
    # Use direct indexing instead of tolist() to avoid memory issues with JAX
    px = int(state.player_position[0])
    py = int(state.player_position[1])

    # Collect nearby interesting tiles
    tiles = _iter_non_grass_tiles(state)

    # Sort by Manhattan distance to the player
    def manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    tiles_sorted = sorted(
        tiles,
        key=lambda t: manhattan((px, py), t[1]),
    )

    # Take up to k unique object types: for each type, show the nearest instance
    seen_names = set()
    closest = []
    for name, pos in tiles_sorted:
        if name not in seen_names:
            seen_names.add(name)
            closest.append((name, pos))
            if len(closest) >= k:
                break

    lines = [f"You in [{px}, {py}]"]
    for name, (x, y) in closest:
        lines.append(f"{name} in [{x}, {y}]")

    lines.append("")
    lines.append("The rest is grass. (You can walk only on grass or path)")

    return "\n".join(lines)


def build_active_agent_prompt(
    state: EnvState,
    goal: str,
    message_from_operator: str,
    previous_actions_analysis: str = "",
) -> str:
    """Generate the full ActiveAgent prompt for a given goal."""
    inventory = format_inventory_from_state(state)
    return generate_agent_prompt(
        goal=goal,
        observation=state,
        message_from_operator=message_from_operator,
        inventory=inventory,
        previous_actions_analysis=previous_actions_analysis,
    )


def do_actions(
    actions: List[str],
    env: Any,
    state: EnvState,
    rngs: jax.Array,
) -> Tuple[Any, EnvState, float, bool, Dict[str, Any], jax.Array]:
    """
    Convert string actions to Craftax `Action` values and apply them.

    The input strings are expected to look like "UP", "DOWN", "LEFT", "RIGHT",
    "DO", etc. Any tokens that do not correspond to a valid `Action` member
    (e.g. commentary words like "(TO", "GATHER", "SOMETHING)") are ignored.
    """
    last_transition: Tuple[Any, EnvState, float, bool, Dict[str, Any]] | None = None

    for action_str in actions:
        token = action_str.strip().upper()

        # Skip empty / non‑action tokens
        if not token:
            continue

        # Only keep tokens that map to a valid Action enum member
        if token not in Action.__members__:
            continue

        action_value = Action[token].value

        # Step the environment with a fresh RNG key and current state
        rngs, step_key = jax.random.split(rngs)
        obs, state, reward, done, info = env.step(step_key, state, action_value)
        last_transition = (obs, state, reward, done, info)

    if last_transition is None:
        raise ValueError("No valid Craftax actions were provided.")

    # Return the last transition together with the updated RNG key
    obs, state, reward, done, info = last_transition
    return obs, state, reward, done, info, rngs


def ask_oracle(
    question: str,
    oracle: Oracle,
    env_state: EnvState,
) -> str:
    """Ask the Oracle a question on behalf of the agent."""
    return oracle.answer(
        question,
        module_name="answer_code_oracle",
        run_code=True,
        env_state=env_state,
    )


def run_single_instruction(
    env: Any,
    agent: ActiveAgentStatistics,
    oracle: Oracle,
    instruction: str,
    run_id: int,
    seed: int = 0,
    log_dir: Optional[Path] = None,
    experiment: Optional[Any] = None,
) -> Tuple[bool, int]:
    """
    Run a single instruction attempt and return (success, num_steps).
    
    Args:
        env: The Craftax environment
        agent: The ActiveAgent with statistics wrapper
        oracle: The Oracle instance
        instruction: The instruction to complete
        run_id: The run number (for seeding)
        seed: Base seed for randomization
        log_dir: Optional directory to save episode logs
        experiment: Optional Comet ML experiment object for logging
        
    Returns:
        Tuple of (success: bool, num_steps: int)
    """
    # Reset environment with unique seed for this run
    rngs = jax.random.PRNGKey(seed + run_id * 1000)
    rngs, reset_key = jax.random.split(rngs)
    obs, initial_state = env.reset(reset_key)
    state = initial_state
    
    message_from_operator = "Hello! Can I help you?"
    num_steps = 0

    # Reset actions history and consecutive questions count at the start of each run
    agent.reset_for_new_episode()
    clear_episode_notes()

    # Episode log for JSON export
    episode_log: List[Dict[str, Any]] = []
    
    # Initialize log file if logging is enabled
    log_filename: Optional[Path] = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        instruction_safe = instruction.lower().replace(" ", "_")
        log_filename = log_dir / f"episode_{instruction_safe}_run{run_id}_{timestamp}.json"
        # Create initial empty log file
        initial_data = {
            "instruction": instruction,
            "run_id": run_id,
            "success": False,
            "num_steps": 0,
            "timestamp": timestamp,
            "episode": [],
        }
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
        print(f"  Episode log file created: {log_filename}")
    
    def convert_to_json_serializable(obj):
        """Recursively convert JAX arrays and other non-serializable types to Python native types."""
        import jax.numpy as jnp
        
        if isinstance(obj, (jax.Array, jnp.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else np.array(obj).tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            # Try to convert to string as fallback
            return str(obj)
    
    def save_log_step():
        """Helper function to save current episode log to file."""
        if log_filename is not None:
            try:
                success_value = check_instruction_completed(instruction, initial_state, state) if num_steps > 0 else False
                # Ensure success is a plain Python bool, not a JAX array or numpy bool
                if isinstance(success_value, (np.bool_, np.bool)):
                    success_value = bool(success_value)
                elif isinstance(success_value, np.ndarray):
                    success_value = bool(success_value.item() if success_value.size > 0 else False)
                else:
                    success_value = bool(success_value)
                
                episode_data = {
                    "instruction": instruction,
                    "run_id": run_id,
                    "success": success_value,
                    "num_steps": int(num_steps),
                    "timestamp": timestamp,
                    "episode": convert_to_json_serializable(episode_log),
                }
                # Convert entire dict to ensure no JAX arrays remain
                episode_data_clean = convert_to_json_serializable(episode_data)
                
                with open(log_filename, "w", encoding="utf-8") as f:
                    json.dump(episode_data_clean, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"  Warning: Failed to save log step: {e}")
                # Try to save at least basic info
                try:
                    basic_data = {
                        "instruction": str(instruction),
                        "run_id": int(run_id),
                        "success": False,
                        "num_steps": int(num_steps),
                        "timestamp": str(timestamp),
                        "episode": [],
                        "error": str(e),
                    }
                    with open(log_filename, "w", encoding="utf-8") as f:
                        json.dump(basic_data, f, indent=2, ensure_ascii=False)
                except:
                    pass  # If even basic save fails, skip it
    
    for step in range(MAX_STEPS_PER_RUN):
        num_steps = step + 1
        
        # Format observation for logging
        observation = format_observation_from_state(state)
        inventory = format_inventory_from_state(state)
        full_observation = f"{observation}\n\n## Your inventory\n{inventory}"

        # Build prompt with current goal and optional previous actions analysis
        previous_analysis = build_previous_actions_analysis(
            agent.actions_history,
            getattr(agent, "consecutive_questions_count", 0),
        )
        prompt = build_active_agent_prompt(
            state, instruction, message_from_operator,
            previous_actions_analysis=previous_analysis,
        )
        
        gen_cfg = GenConfig(max_new_tokens=64, do_sample=False)
        
        # Get agent response (exactly as in server.py)
        raw_answer = agent.chat(
            user_message=prompt,
            system_message="You are the agent described in the prompt.",
            gen=gen_cfg,
        )
        
        # Parse answer using the same function as server
        parsed = parse_agent_answer(raw_answer)
        
        # Debug logging for unparseable answers
        if not parsed:
            print(f"  Warning: Unparseable answer at step {num_steps}")
            print(f"  Raw answer type: {type(raw_answer)}")
            print(f"  Raw answer length: {len(raw_answer) if raw_answer else 0}")
            if raw_answer:
                print(f"  Raw answer (first 500 chars): {repr(raw_answer[:500])}")
                print(f"  Contains '--- Act ---': {'--- Act ---' in raw_answer}")
                print(f"  Contains '--- Q ---': {'--- Q ---' in raw_answer}")
        
        # Initialize step log entry
        step_log: Dict[str, Any] = {
            "step": num_steps,
            "observation": full_observation,
            "raw_answer": str(raw_answer)[:1000] if raw_answer else "",  # Save raw answer for debugging
            "action": None,
            "oracle_response": None,
        }
        
        if "action" in parsed:
            # Agent wants to take action (exactly as in server.py)
            action_str = parsed["action"]
            step_log["action"] = action_str
            agent.record_action(action_str)

            craftax_actions = action_str.split()
            obs, state, reward, done, info, rngs = do_actions(
                craftax_actions, env, state, rngs
            )
            
            # Log step to Comet if available
            if experiment is not None:
                instruction_safe = instruction.lower().replace(" ", "_")
                experiment.log_metric(
                    f"{instruction_safe}/run_{run_id}/step_{num_steps}/reward",
                    float(reward),
                    step=num_steps
                )
            
            # Check if instruction is completed
            if check_instruction_completed(instruction, initial_state, state):
                step_log["instruction_completed"] = True
                episode_log.append(step_log)
                success = True
                # Log success to Comet
                if experiment is not None:
                    instruction_safe = instruction.lower().replace(" ", "_")
                    experiment.log_metric(
                        f"{instruction_safe}/run_{run_id}/completed",
                        1,
                        step=num_steps
                    )
                save_log_step()  # Save final state
                break
            
            # Don't exit early if done=True - continue until MAX_STEPS_PER_RUN
            # The environment might mark episode as done for various reasons,
            # but we want to give the agent full 50 iterations to complete the task
            if done:
                # Episode marked as done, but continue trying until max steps
                # Log this but don't exit - give agent full 50 iterations
                if num_steps < MAX_STEPS_PER_RUN:
                    print(f"  Note: Episode marked as done at step {num_steps}, but continuing to step {MAX_STEPS_PER_RUN}...")
                step_log["episode_done"] = True
                
        elif "question" in parsed:
            # Agent asks a question - record it and clear actions history so "Previous actions analysis" is updated
            agent.record_question()
            agent.clear_actions_history()
            # Route to Oracle (exactly as in server.py)
            question = parsed["question"]
            step_log["action"] = f"QUESTION: {question}"
            message_from_operator = ask_oracle(question, oracle, state)
            step_log["oracle_response"] = message_from_operator
            # Continue loop with Oracle's answer
            
        else:
            # Unparseable answer - log but continue trying
            # Don't fail immediately, give agent more chances
            print(f"  Warning: Unparseable answer at step {num_steps}, continuing...")
            step_log["action"] = f"UNPARSEABLE: {str(raw_answer)[:200]}"  # Truncate long responses
            # Optionally, we could reset message_from_operator to encourage retry
            # message_from_operator = "Please provide a valid action or question."
            # For now, just continue the loop
        
        episode_log.append(step_log)
        # Save log after each step
        save_log_step()
        
        # Reset message_from_operator for next iteration (unless it was set by Oracle)
        if "question" not in parsed:
            message_from_operator = ""
    
    # Determine success
    success = check_instruction_completed(instruction, initial_state, state) if num_steps > 0 else False
    
    # Log final result to Comet
    if experiment is not None:
        instruction_safe = instruction.lower().replace(" ", "_")
        experiment.log_metric(
            f"{instruction_safe}/run_{run_id}/final_success",
            1 if success else 0,
            step=num_steps
        )
        experiment.log_metric(
            f"{instruction_safe}/run_{run_id}/final_steps",
            num_steps,
            step=num_steps
        )
    
    # Final save to ensure everything is up to date
    if log_filename is not None:
        save_log_step()
        print(f"  Episode log finalized: {log_filename}")
    
    # Max steps reached without completing instruction
    return success, num_steps


def _log_comet_bar_charts(
    experiment: Any,
    all_results: List[Dict[str, Any]],
    step: int,
) -> None:
    """
    Create and log bar charts for success rate and avg steps to Comet ML.
    """
    if not all_results or experiment is None:
        return
    try:
        instructions = [r["instruction"] for r in all_results]
        success_rates = [r["success_rate"] * 100 for r in all_results]
        avg_steps = [r["avg_steps"] for r in all_results]

        # Success rate bar chart
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        bars1 = ax1.bar(range(len(instructions)), success_rates, color="steelblue", edgecolor="black")
        ax1.set_xticks(range(len(instructions)))
        ax1.set_xticklabels(instructions, rotation=45, ha="right")
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Success Rate by Instruction")
        ax1.set_ylim(0, 105)
        for bar, val in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        experiment.log_figure(figure=fig1, figure_name="success_rate_bar_chart", step=step)
        plt.close(fig1)

        # Avg steps bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        bars2 = ax2.bar(range(len(instructions)), avg_steps, color="coral", edgecolor="black")
        ax2.set_xticks(range(len(instructions)))
        ax2.set_xticklabels(instructions, rotation=45, ha="right")
        ax2.set_ylabel("Average Steps")
        ax2.set_title("Average Steps by Instruction")
        for bar, val in zip(bars2, avg_steps):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        experiment.log_figure(figure=fig2, figure_name="avg_steps_bar_chart", step=step)
        plt.close(fig2)
    except Exception as e:
        print(f"Warning: Failed to log bar charts to Comet: {e}")


def benchmark_instruction(
    env: Any,
    agent: ActiveAgentStatistics,
    oracle: Oracle,
    instruction: str,
    num_runs: int = NUM_RUNS_PER_INSTRUCTION,
    experiment: Optional[Any] = None,
    log_dir: Optional[Path] = None,
    episode_counter: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Benchmark a single instruction multiple times.
    
    Args:
        env: The Craftax environment
        agent: The ActiveAgent with statistics wrapper
        oracle: The Oracle instance
        instruction: The instruction to benchmark
        num_runs: Number of runs per instruction
        experiment: Optional Comet ML experiment object for logging
        log_dir: Optional directory to save episode logs
        episode_counter: Optional mutable list [int] for global episode step (for Comet logging)
        
    Returns a dict with:
        - instruction: str
        - num_runs: int
        - successes: int
        - success_rate: float
        - avg_steps: float
        - results: List[Tuple[bool, int]]  # (success, num_steps) for each run
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {instruction}")
    print(f"{'='*60}")
    
    results: List[Tuple[bool, int]] = []
    
    for run_id in range(num_runs):
        print(f"\nRun {run_id + 1}/{num_runs}...")
        success, num_steps = run_single_instruction(
            env, agent, oracle, instruction, run_id, log_dir=log_dir, experiment=experiment
        )
        results.append((success, num_steps))
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status} (steps: {num_steps})")

        # Log results after each episode to Comet (with global episode step)
        if experiment is not None:
            ep_step = episode_counter[0] if episode_counter is not None else run_id
            instruction_safe = instruction.lower().replace(" ", "_")
            experiment.log_metrics(
                {
                    "episode/success": 1 if success else 0,
                    "episode/steps": num_steps,
                    f"episode/{instruction_safe}/success": 1 if success else 0,
                    f"episode/{instruction_safe}/steps": num_steps,
                },
                step=ep_step,
            )
            # Running success rate and avg steps so far for this instruction
            running_successes = sum(1 for s, _ in results if s)
            running_avg_steps = np.mean([s for _, s in results])
            experiment.log_metrics(
                {
                    f"episode/{instruction_safe}/running_success_rate": running_successes / len(results),
                    f"episode/{instruction_safe}/running_avg_steps": running_avg_steps,
                },
                step=ep_step,
            )
            if episode_counter is not None:
                episode_counter[0] += 1
    
    successes = sum(1 for success, _ in results if success)
    success_rate = successes / num_runs
    avg_steps = np.mean([steps for _, steps in results])
    
    print(f"\nResults for '{instruction}':")
    print(f"  Success Rate (SR): {success_rate:.2%} ({successes}/{num_runs})")
    print(f"  Average Steps: {avg_steps:.1f}")
    
    # Log aggregated metrics to Comet
    # Use step=instruction_index to create line charts comparing instructions
    if experiment is not None:
        instruction_safe = instruction.lower().replace(" ", "_")
        # Get instruction index for step parameter (for line charts comparing instructions)
        instruction_idx = INSTRUCTIONS.index(instruction) if instruction in INSTRUCTIONS else 0
        # Log with step to create line charts for comparison across instructions
        experiment.log_metric(f"{instruction_safe}/success_rate", success_rate, step=instruction_idx)
        experiment.log_metric(f"{instruction_safe}/avg_steps", avg_steps, step=instruction_idx)
        experiment.log_metric(f"{instruction_safe}/successes", successes, step=instruction_idx)
        experiment.log_metric(f"{instruction_safe}/num_runs", num_runs, step=instruction_idx)
        
        # Also log without step for bar charts (single value per instruction)
        experiment.log_metric(f"summary/{instruction_safe}/success_rate", success_rate)
        experiment.log_metric(f"summary/{instruction_safe}/avg_steps", avg_steps)
    
    return {
        "instruction": instruction,
        "num_runs": num_runs,
        "successes": successes,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "results": results,
    }


def main() -> None:
    """Entry point for running the benchmarking suite."""
    print("="*60)
    print("Oracle - Active Agent Interactions Performance Benchmarking")
    print("="*60)
    
    # Initialize Comet ML experiment (required)
    if not COMET_AVAILABLE:
        raise RuntimeError("Comet ML not available. Install with: pip install comet_ml")
    try:
        experiment = Experiment(
            project_name="craftax-oracle-benchmarking",
            workspace=None,  # Will use default workspace from config/env
            auto_output_logging="native",  # Enable automatic logging
            log_graph=False,
        )
        print("Comet ML experiment initialized")
        print(f"  Experiment key: {experiment.get_key()}")
        print(f"  Project URL: {experiment.url}")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Comet ML: {e}") from e
    
    # Load Oracle configuration
    config_path = Path(__file__).resolve().parent.parent / "config" / "oracle_config.yaml"
    config = load_config(path=str(config_path))
    oracle = Oracle(config)
    
    # Create environment
    env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
    
    # Create agent with statistics wrapper
    model_name = "qwen/qwen3-next-80b-a3b-instruct"
    base_agent = ActiveAgent(
        model_name=model_name,
        reasoning=False,
    )
    agent = ActiveAgentStatistics(base_agent)
    
    # Create log directory for episode logs
    log_dir = Path(__file__).resolve().parent / "episode_logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = log_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Episode logs will be saved to: {log_dir}")
    
    # Log hyperparameters to Comet
    if experiment is not None:
        experiment.log_parameters({
            "model_name": model_name,
            "num_runs_per_instruction": NUM_RUNS_PER_INSTRUCTION,
            "max_steps_per_run": MAX_STEPS_PER_RUN,
            "instructions": INSTRUCTIONS,
            "num_instructions": len(INSTRUCTIONS),
        })
        experiment.log_parameter("environment", "Craftax-Classic-Symbolic-v1")
        experiment.log_parameter("log_dir", str(log_dir))
    
    # Benchmark each instruction
    all_results: List[Dict[str, Any]] = []
    episode_counter: List[int] = [0]

    for instruction_idx, instruction in enumerate(INSTRUCTIONS):
        result = benchmark_instruction(
            env, agent, oracle, instruction, NUM_RUNS_PER_INSTRUCTION,
            experiment, log_dir, episode_counter=episode_counter,
        )
        all_results.append(result)

        # Log bar charts after each instruction
        if experiment is not None:
            experiment.log_metric("progress/instruction_index", instruction_idx + 1)
            _log_comet_bar_charts(experiment, all_results, step=episode_counter[0] - 1)
    
    # Calculate overall statistics
    overall_success_rate = np.mean([r["success_rate"] for r in all_results])
    overall_avg_steps = np.mean([r["avg_steps"] for r in all_results])
    total_successes = sum(r["successes"] for r in all_results)
    total_runs = sum(r["num_runs"] for r in all_results)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Instruction':<30} {'SR':<10} {'Avg Steps':<12} {'Successes'}")
    print("-" * 60)
    
    for result in all_results:
        print(
            f"{result['instruction']:<30} "
            f"{result['success_rate']:>6.2%}   "
            f"{result['avg_steps']:>8.1f}   "
            f"{result['successes']}/{result['num_runs']}"
        )
    
    print("-" * 60)
    print(
        f"{'OVERALL':<30} "
        f"{overall_success_rate:>6.2%}   "
        f"{overall_avg_steps:>8.1f}   "
        f"{total_successes}/{total_runs}"
    )
    
    # Log overall metrics to Comet
    # These are final summary metrics, no step needed (will show as single values)
    if experiment is not None:
        experiment.log_metric("overall/success_rate", overall_success_rate)
        experiment.log_metric("overall/avg_steps", overall_avg_steps)
        experiment.log_metric("overall/total_successes", total_successes)
        experiment.log_metric("overall/total_runs", total_runs)
        # Final bar charts
        _log_comet_bar_charts(experiment, all_results, step=episode_counter[0])
    
    # Print agent statistics
    print("\n" + "="*60)
    print("AGENT STATISTICS")
    print("="*60)
    print(f"Total calls: {agent.total_calls}")
    print(f"Questions asked: {agent.questions}")
    print(f"Actions taken: {agent.actions}")
    print(f"Failures: {agent.failures}")
    
    # Log agent statistics to Comet
    if experiment is not None:
        experiment.log_metrics({
            "agent/total_calls": agent.total_calls,
            "agent/questions": agent.questions,
            "agent/actions": agent.actions,
            "agent/failures": agent.failures,
        })
        if agent.total_calls > 0:
            experiment.log_metric("agent/question_rate", agent.questions / agent.total_calls)
            experiment.log_metric("agent/action_rate", agent.actions / agent.total_calls)
            experiment.log_metric("agent/failure_rate", agent.failures / agent.total_calls)
    
    # Print Oracle statistics if available
    print("\n" + "="*60)
    print("BENCHMARKING COMPLETE")
    print("="*60)
    
    # End Comet experiment
    if experiment is not None:
        experiment.end()
        print("Comet ML experiment ended")


if __name__ == "__main__":
    main()
