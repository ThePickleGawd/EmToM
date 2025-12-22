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
    parser.add_argument("--config-name", default="examples/emtom_two_robots")
    args = parser.parse_args()

    # Load task
    try:
        with open(args.task_file) as f:
            task_data = json.load(f)
    except Exception as e:
        print(json.dumps({"valid": False, "error": f"Failed to load task: {e}"}))
        sys.exit(1)

    golden = task_data.get("golden_trajectory", [])
    if not golden:
        print(json.dumps({"valid": False, "error": "No golden_trajectory found"}))
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
        print(json.dumps({"valid": False, "error": f"Import error: {e}"}))
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
        print(json.dumps({"valid": False, "error": f"Config error: {e}"}))
        sys.exit(1)

    # Convert to GeneratedTask
    try:
        task = GeneratedTask.from_dict(task_data)
    except Exception as e:
        print(json.dumps({"valid": False, "error": f"Invalid task format: {e}"}))
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

        task_mechanics = {
            "mechanics": [
                {"mechanic_type": b.mechanic_type, **b.to_dict()}
                for b in task.mechanic_bindings
            ]
        } if task.mechanic_bindings else None

        runner.setup(
            env_interface=env_interface,
            task_data=task_mechanics,
            output_dir="/tmp/emtom_verify",
            task=task,
            save_video=False,
        )
    except Exception as e:
        print(json.dumps({"valid": False, "error": f"Environment setup failed: {e}"}))
        sys.exit(1)

    # Execute trajectory
    executed_steps = []
    print(f"\n=== Executing Golden Trajectory ({len(golden)} steps) ===", file=sys.stderr)
    try:
        for i, step in enumerate(golden):
            agent_str = step.get("agent", "agent_0")
            agent_id = int(agent_str.split("_")[1])
            action = step.get("action")
            target = step.get("target")
            message = step.get("message")

            # Handle Place action - needs 5 comma-separated arguments
            # Support both formats:
            # 1. Full format: {"target": "obj, on, furniture, None, None"}
            # 2. Legacy format: {"target": "obj", "receptacle": "furniture"}
            if action == "Place" and target:
                # Check if target already has commas (full format)
                if "," not in str(target):
                    # Legacy format - build the full argument string
                    receptacle = step.get("receptacle", "")
                    spatial_relation = step.get("spatial_relation", "on")
                    spatial_constraint = step.get("spatial_constraint", "None")
                    reference_object = step.get("reference_object", "None")
                    target = f"{target}, {spatial_relation}, {receptacle}, {spatial_constraint}, {reference_object}"

            # Skip Communicate
            if action == "Communicate":
                print(f"  Step {i+1}: {agent_str} Communicate(\"{message[:50]}...\") [SKIP]", file=sys.stderr)
                executed_steps.append({
                    "step": i, "agent": agent_str, "action": action,
                    "message": message, "success": True, "skipped": True
                })
                continue

            print(f"  Step {i+1}: {agent_str} {action}({target})...", file=sys.stderr, end=" ")
            result = runner.execute_action(uid=agent_id, action_name=action, target=target)
            success = result.get("success", False)
            obs = result.get("observation", "")[:100]

            if success:
                print(f"OK", file=sys.stderr)
            else:
                print(f"FAILED: {obs}", file=sys.stderr)

            executed_steps.append({
                "step": i, "agent": agent_str, "action": action,
                "target": target, "success": success,
                "observation": result.get("observation", "")[:200]
            })

            if not success:
                runner.cleanup()
                print(json.dumps({
                    "valid": False,
                    "failed_step": i,
                    "action": f"{agent_str}: {action}({target})",
                    "error": result.get("observation", "Action failed"),
                    "executed_steps": executed_steps,
                }, indent=2))
                sys.exit(0)

        # Evaluate success
        print(f"\n=== Evaluating Success Condition ===", file=sys.stderr)
        evaluation = runner.evaluate_task()
        success_met = evaluation.get("success", False)
        runner.cleanup()

        if success_met:
            print(f"  Result: SUCCESS", file=sys.stderr)
            print(json.dumps({
                "valid": True,
                "steps_executed": len(executed_steps),
                "success_condition_met": True,
                "executed_steps": executed_steps,
            }, indent=2))
        else:
            print(f"  Result: FAILED", file=sys.stderr)
            print(f"  Reason: {evaluation.get('failure_explanations', ['Unknown'])}", file=sys.stderr)
            print(json.dumps({
                "valid": False,
                "error": "Success condition not met after trajectory",
                "evaluation": evaluation,
                "executed_steps": executed_steps,
            }, indent=2))

    except Exception as e:
        runner.cleanup()
        print(json.dumps({
            "valid": False,
            "error": f"Verification error: {e}",
            "executed_steps": executed_steps,
        }, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
