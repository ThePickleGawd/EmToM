#!/usr/bin/env python3
"""
Standalone script to verify golden trajectory.

This runs as a subprocess to get a fresh GL context.

Usage:
    python emtom/task_gen/verify_trajectory.py --task-file <path> --config-name <config>
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Verify golden trajectory")
    parser.add_argument("--task-file", required=True, help="Path to task JSON")
    parser.add_argument("--result-file", required=True, help="Path to write result JSON")
    parser.add_argument("--config-name", default="examples/emtom_2_robots")
    args = parser.parse_args()

    def write_result(result: dict):
        """Write result to file instead of stdout."""
        with open(args.result_file, 'w') as f:
            json.dump(result, f, indent=2)

    # Load task
    try:
        with open(args.task_file) as f:
            task_data = json.load(f)
    except Exception as e:
        write_result({"valid": False, "error": f"Failed to load task: {e}"})
        sys.exit(1)

    golden = task_data.get("golden_trajectory", [])
    if not golden:
        write_result({"valid": False, "error": "No golden_trajectory found"})
        sys.exit(1)

    # Import and setup Habitat
    try:
        import hydra
        from omegaconf import DictConfig
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        from habitat_llm.agent.env import register_actions, register_measures, register_sensors
        from habitat_llm.agent.env.dataset import CollaborationDatasetV0
        from habitat_llm.agent.env.environment_interface import EnvironmentInterface
        from habitat_llm.utils import fix_config, setup_config

        from emtom.runner.verification import VerificationRunner
        from emtom.task_gen import GeneratedTask
    except ImportError as e:
        write_result({"valid": False, "error": f"Import error: {e}"})
        sys.exit(1)

    # Initialize Hydra config
    try:
        from omegaconf import open_dict, OmegaConf

        GlobalHydra.instance().clear()
        config_dir = str(project_root / "habitat_llm" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        config = compose(config_name=args.config_name)

        # Manually override Hydra interpolations BEFORE fix_config tries to resolve them
        # These would fail with "HydraConfig was not set" otherwise
        output_dir = "/tmp/emtom_verify"
        with open_dict(config):
            # Override evaluation.output_dir (uses ${hydra:runtime.output_dir})
            if "evaluation" in config:
                config.evaluation.output_dir = output_dir

            # Override paths that use ${hydra:runtime.output_dir}
            if "paths" in config:
                config.paths.results_dir = f"{output_dir}/results"
                config.paths.epi_result_file_path = f"{output_dir}/results/episode_result_log.csv"
                config.paths.run_result_file_path = f"{output_dir}/results/run_result_log.csv"
                config.paths.end_result_file_path = f"{output_dir}/results/end_result_log.csv"

        fix_config(config)
        config = setup_config(config, seed=47668090)
    except Exception as e:
        write_result({"valid": False, "error": f"Config error: {e}"})
        sys.exit(1)

    # Convert to GeneratedTask
    try:
        task = GeneratedTask.from_dict(task_data)
    except Exception as e:
        write_result({"valid": False, "error": f"Invalid task format: {e}"})
        sys.exit(1)

    # Setup environment
    try:
        register_sensors(config)
        register_actions(config)
        register_measures(config)

        dataset = CollaborationDatasetV0(config.habitat.dataset)
        env_interface = EnvironmentInterface(config, dataset=dataset, init_wg=False)

        # Load the specific episode from the task
        dataset_episode_id = task.dataset_episode_id
        print(f"Loading episode: {dataset_episode_id} (scene: {task.scene_id})", file=sys.stderr)
        env_interface.reset_environment(episode_id=dataset_episode_id)

        runner = VerificationRunner(config)

        # Build task data with mechanics, items, and locked_containers
        task_mechanics = {}
        if task.mechanic_bindings:
            task_mechanics["mechanics"] = [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        if task.items:
            task_mechanics["items"] = task.items  # Already list of dicts
        if task.locked_containers:
            task_mechanics["locked_containers"] = task.locked_containers
        task_mechanics = task_mechanics if task_mechanics else None

        runner.setup(
            env_interface=env_interface,
            task_data=task_mechanics,
            output_dir="/tmp/emtom_verify",
            task=task,
            save_video=False,
        )
    except Exception as e:
        write_result({"valid": False, "error": f"Environment setup failed: {e}"})
        sys.exit(1)

    # Execute trajectory
    # New format: each step has "actions" array with all agents' actions for that step
    executed_steps = []
    print(f"\n=== Executing Golden Trajectory ({len(golden)} steps) ===", file=sys.stderr)
    try:
        for step_idx, step in enumerate(golden):
            # Get the actions for this step
            actions = step.get("actions", [])
            if not actions:
                print(f"  Step {step_idx+1}: No actions found in step, skipping", file=sys.stderr)
                continue

            print(f"  Step {step_idx+1}:", file=sys.stderr)
            step_results = []

            for action_entry in actions:
                agent_str = action_entry.get("agent", "agent_0")
                agent_id = int(agent_str.split("_")[1])
                action = action_entry.get("action")
                target = action_entry.get("target")
                message = action_entry.get("message")

                # Skip Wait actions
                if action == "Wait":
                    print(f"    {agent_str}: Wait [SKIP]", file=sys.stderr)
                    step_results.append({
                        "agent": agent_str, "action": action,
                        "success": True, "skipped": True
                    })
                    continue

                # Handle Place action - needs 5 comma-separated arguments
                # Support both formats:
                # 1. Full format: {"target": "obj, on, furniture, None, None"}
                # 2. Legacy format: {"target": "obj", "receptacle": "furniture"}
                if action == "Place" and target:
                    # Check if target already has commas (full format)
                    if "," not in str(target):
                        # Legacy format - build the full argument string
                        receptacle = action_entry.get("receptacle", "")
                        spatial_relation = action_entry.get("spatial_relation", "on")
                        spatial_constraint = action_entry.get("spatial_constraint", "None")
                        reference_object = action_entry.get("reference_object", "None")
                        target = f"{target}, {spatial_relation}, {receptacle}, {spatial_constraint}, {reference_object}"

                # Skip Communicate
                if action == "Communicate":
                    msg_preview = message[:50] if message else ""
                    print(f"    {agent_str}: Communicate(\"{msg_preview}...\") [SKIP]", file=sys.stderr)
                    step_results.append({
                        "agent": agent_str, "action": action,
                        "message": message, "success": True, "skipped": True
                    })
                    continue

                result = runner.execute_action(uid=agent_id, action_name=action, target=target)
                success = result.get("success", False)
                obs = result.get("observation", "")

                # Print action + observation
                status = "✓" if success else "✗"
                print(f"    {agent_str}: {action}[{target}] {status}", file=sys.stderr)
                if obs:
                    print(f"      → {obs}", file=sys.stderr)

                step_results.append({
                    "agent": agent_str, "action": action,
                    "target": target, "success": success,
                    "observation": result.get("observation", "")[:200]
                })

                if not success:
                    runner.cleanup()
                    write_result({
                        "valid": False,
                        "failed_step": step_idx,
                        "action": f"{agent_str}: {action}({target})",
                        "error": result.get("observation", "Action failed"),
                        "executed_steps": executed_steps + [{"step": step_idx, "actions": step_results}],
                    })
                    sys.exit(0)

            executed_steps.append({"step": step_idx, "actions": step_results})

        # Evaluate success
        print(f"\n=== Evaluating Success Condition ===", file=sys.stderr)
        evaluation = runner.evaluate_task()
        success_met = evaluation.get("success", False)
        runner.cleanup()

        if success_met:
            print(f"  Result: SUCCESS", file=sys.stderr)
            write_result({
                "valid": True,
                "steps_executed": len(executed_steps),
                "success_condition_met": True,
                "executed_steps": executed_steps,
            })
        else:
            print(f"  Result: FAILED", file=sys.stderr)
            print(f"  Reason: {evaluation.get('failure_explanations', ['Unknown'])}", file=sys.stderr)
            write_result({
                "valid": False,
                "error": "Success condition not met after trajectory",
                "evaluation": evaluation,
                "executed_steps": executed_steps,
            })

    except Exception as e:
        runner.cleanup()
        write_result({
            "valid": False,
            "error": f"Verification error: {e}",
            "executed_steps": executed_steps,
        })
        sys.exit(1)


if __name__ == "__main__":
    main()
