#!/usr/bin/env python3
"""
Generate Hydra config files for multi-agent EMTOM configurations.

This script generates all the necessary config files to support N agents (up to 10).
Run once to generate configs, then commit them.

Usage:
    python scripts/generate_multi_agent_configs.py
"""

from pathlib import Path


def generate_habitat_conf(n_agents: int) -> str:
    """Generate emtom_spot_multi_agent_N.yaml content."""

    # Header and defaults
    lines = [
        "# @package _global_",
        "",
        f"# {n_agents} identical Spot robots, each with head, arm, and jaw cameras.",
        "defaults:",
        "  - /habitat: habitat_config_base",
    ]

    # Agent defaults
    for i in range(n_agents):
        lines.append(f"  - /habitat_conf/habitat_agent@habitat.simulator.agents.agent_{i}: rgbd_head_rgbd_arm_rgbd_jaw_agent_vis")

    lines.extend([
        "  - /habitat_conf/dataset: collaboration_hssd",
        "",
        "  - /habitat/task/lab_sensors:",
        "    - humanoid_detector_sensor",
        "  - _self_",
        "",
        "habitat:",
        "  gym:",
        "    obs_keys:",
    ])

    # Observation keys for each agent
    obs_keys = [
        "third_rgb",
        "articulated_agent_arm_depth",
        "articulated_agent_arm_rgb",
        "articulated_agent_arm_panoptic",
        "head_depth",
        "head_rgb",
        "head_panoptic",
        "relative_resting_position",
        "joint",
        "ee_pos",
        "is_holding",
        "humanoid_detector_sensor",
        "articulated_agent_jaw_rgb",
        "articulated_agent_jaw_depth",
        "articulated_agent_jaw_panoptic",
    ]

    for i in range(n_agents):
        lines.append(f"      # Agent {i}")
        for key in obs_keys:
            lines.append(f"      - agent_{i}_{key}")
        lines.append("")

    # Task and simulator config
    lines.extend([
        "  task:",
        "    lab_sensors:",
        "      humanoid_detector_sensor:",
        "        return_image: False",
        "        is_return_image_bbox: False",
        "  environment:",
        "    max_episode_steps: 1000",
        "  simulator:",
        "    type: CollaborationSim-v0",
        "    additional_object_paths:",
        '      - "data/objects/ycb/configs/"',
        '      - "data/objects_ovmm/train_val/ai2thorhab/configs/objects"',
        '      - "data/objects_ovmm/train_val/amazon_berkeley/configs"',
        '      - "data/objects_ovmm/train_val/google_scanned/configs"',
        '      - "data/objects_ovmm/train_val/hssd/configs/objects"',
        "    concur_render: False",
        "    auto_sleep: True",
        "    agents:",
    ])

    # Simulator agents
    for i in range(n_agents):
        lines.extend([
            f"      agent_{i}:",
            "        radius: 0.3",
            "        articulated_agent_urdf: ./data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf",
            "        articulated_agent_type: SpotRobot",
            "        joint_start_noise: 0.0",
        ])

    lines.extend([
        "    kinematic_mode: True",
        "    step_physics: False",
        "    habitat_sim_v0:",
        "      allow_sliding: True",
        "      enable_physics: True",
    ])

    return "\n".join(lines) + "\n"


def generate_task_conf(n_agents: int) -> str:
    """Generate rearrange_easy_multi_agent_N.yaml content."""

    lines = [
        "# @package habitat.task",
        "",
        "defaults:",
        "  - /habitat/task: task_config_base",
        "  - /habitat/task/actions@actions.agent_0_arm_action: arm_action",
    ]

    # Base velocity actions for all agents
    for i in range(n_agents):
        lines.append(f"  - /habitat/task/actions@actions.agent_{i}_base_velocity: base_velocity")

    lines.extend([
        "  - /habitat/task/measurements:",
        "    - articulated_agent_force",
        "    - force_terminate",
        "    - articulated_agent_colls",
        "    - zero",
        "    - num_steps",
        "    - did_violate_hold_constraint",
        "    - gfx_replay_measure",
        "  - /habitat/task/lab_sensors:",
        "    - relative_resting_pos_sensor",
        "    - target_start_sensor",
        "    - goal_sensor",
        "    - joint_sensor",
        "    - end_effector_sensor",
        "    - is_holding_sensor",
        "    - end_effector_sensor",
        "    - target_start_gps_compass_sensor",
        "    - target_goal_gps_compass_sensor",
        "  - _self_",
        "",
        "",
        "type: RearrangeEmptyTask-v0",
        'reward_measure: "zero"',
        'success_measure: "zero"',
        "success_reward: 100.0",
        "slack_reward: -0.01",
        "end_on_success: True",
        "constraint_violation_ends_episode: False",
        "constraint_violation_drops_object: True",
        "task_spec: rearrange_easy",
        "",
        "measurements:",
        "  force_terminate:",
        "    max_accum_force: -1",
        "    max_instant_force: -1",
    ])

    return "\n".join(lines) + "\n"


def generate_evaluation_conf(n_agents: int) -> str:
    """Generate decentralized_evaluation_runner_N_agent_openai.yaml content."""

    lines = [
        "defaults:",
    ]

    # Agent configs
    for i in range(n_agents):
        lines.append(f"  - /agent@agents.agent_{i}.config: oracle_rearrange_agent")

    lines.append("")

    # Planner configs
    for i in range(n_agents):
        lines.append(f"  - /planner@agents.agent_{i}.planner: llm_planner_openai")

    lines.extend([
        "",
        'type: "decentralized"',
        "truncate_length: 50  # max characters in file name, long files names can raise OS errors",
        "save_video: True",
        "output_dir: ${hydra:runtime.output_dir}",
        "",
        "agents:",
    ])

    # Agent UIDs
    for i in range(n_agents):
        lines.extend([
            f"  agent_{i}:",
            f"    uid: {i}",
        ])

    return "\n".join(lines) + "\n"


def generate_example_robots_conf(n_agents: int) -> str:
    """Generate emtom_N_robots.yaml content."""

    agent_names = ", ".join([f"agent_{i}" for i in range(n_agents)])
    camera_prefixes = ", ".join(["articulated_agent_jaw"] * n_agents)

    lines = [
        "# @package _global_",
        "",
        f"# EMTOM {n_agents}-robot preset (all embodied as Spot robots)",
        "defaults:",
        "  - /training@             : base_train",
        f"  - /habitat_conf          : emtom_spot_multi_agent_{n_agents}",
        f"  - /habitat_conf/task     : rearrange_easy_multi_agent_{n_agents}",
        f"  - /evaluation: decentralized_evaluation_runner_{n_agents}_agent_openai",
        "  - /world_model@world_model: gt_graph",
        "  - /trajectory@trajectory : trajectory_logger",
        "  - /agent/@agents.agent_0.config: oracle_rearrange_agent",
    ]

    # Agent configs (agent_1+ use object_states_agent)
    for i in range(1, n_agents):
        lines.append(f"  - /agent/@agents.agent_{i}.config: oracle_rearrange_object_states_agent")

    lines.extend([
        f"  # Planners for all agents come from decentralized_evaluation_runner_{n_agents}_agent_openai",
        "  - /wandb_conf@                : own",
        "  - override /hydra/output : output",
        "  # Override planner instruct to use EMTOM-specific prompts (not PARTNR few-shot examples)",
    ])

    # Instruct overrides
    for i in range(n_agents):
        lines.append(f"  - override /instruct@evaluation.agents.agent_{i}.planner.plan_config.instruct: emtom_no_visual")

    lines.extend([
        "  - _self_",
        "",
        "world_model:",
        "  partial_obs: False",
        "agent_asymmetry: False",
        "trajectory:",
        f"  agent_names: [{agent_names}]",
        f"  camera_prefixes: [{camera_prefixes}]",
        "",
        "device      : cuda",
        "instruction : ''",
        'mode: "dataset"',
        "env: habitat",
        "num_runs_per_episode: 1",
        "num_proc: 1",
        "dry_run: False",
        "robot_agent_uid: 0",
        "human_agent_uid: 1",
        "",
        "agents:",
    ])

    # Agents section
    for i in range(n_agents):
        lines.extend([
            f"  agent_{i}:",
            f"    uid: {i}",
        ])

    lines.extend([
        "",
        "paths:",
        "  results_dir: ${hydra:runtime.output_dir}/results",
        '  epi_result_file_path: "${paths.results_dir}/episode_result_log.csv"',
        '  run_result_file_path: "${paths.results_dir}/run_result_log.csv}"',
        '  end_result_file_path: "${paths.results_dir}/end_result_log.csv}"',
        "",
        "evaluation:",
        "  do_print: True",
        "  save_video: True",
        "  log_data: True",
        "  save_rgb: False",
        "  log_detailed_traces: True",
        "  agents:",
    ])

    # Evaluation agents
    for i in range(n_agents):
        lines.extend([
            f"    agent_{i}:",
            f"      uid: {i}",
        ])

    lines.extend([
        "",
        "habitat:",
        "  task:",
        "    pddl_domain_def: fp",
        "",
        "hydra:",
        "  job:",
        f"    name: 'emtom_{n_agents}_robots'",
        "    chdir: False",
        "  run:",
        "    dir: outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}-${file_stem:${habitat.dataset.data_path}}",
    ])

    return "\n".join(lines) + "\n"


def generate_example_humans_conf(n_agents: int) -> str:
    """Generate emtom_N_humans.yaml content."""

    agent_names = ", ".join([f"agent_{i}" for i in range(n_agents)])
    camera_prefixes = ", ".join(["head"] * n_agents)

    lines = [
        "# @package _global_",
        "",
        f"# EMTOM {n_agents}-humanoid preset",
        "defaults:",
        "  - /training@             : base_train",
        f"  - /habitat_conf          : emtom_humanoid_multi_agent_{n_agents}",
        f"  - /habitat_conf/task     : rearrange_easy_multi_agent_{n_agents}",
        f"  - /evaluation: decentralized_evaluation_runner_{n_agents}_agent_openai",
        "  - /world_model@world_model: gt_graph",
        "  - /trajectory@trajectory : trajectory_logger",
        "  - /agent/@agents.agent_0.config: oracle_rearrange_agent",
    ]

    for i in range(1, n_agents):
        lines.append(f"  - /agent/@agents.agent_{i}.config: oracle_rearrange_object_states_agent")

    lines.extend([
        f"  # Planners for all agents come from decentralized_evaluation_runner_{n_agents}_agent_openai",
        "  - /wandb_conf@                : own",
        "  - override /hydra/output : output",
        "  # Override planner instruct to use EMTOM-specific prompts (not PARTNR few-shot examples)",
    ])

    for i in range(n_agents):
        lines.append(f"  - override /instruct@evaluation.agents.agent_{i}.planner.plan_config.instruct: emtom_no_visual")

    lines.extend([
        "  - _self_",
        "",
        "world_model:",
        "  partial_obs: False",
        "agent_asymmetry: False",
        "trajectory:",
        f"  agent_names: [{agent_names}]",
        f"  camera_prefixes: [{camera_prefixes}]",
        "",
        "device      : cuda",
        "instruction : ''",
        'mode: "dataset"',
        "env: habitat",
        "num_runs_per_episode: 1",
        "num_proc: 1",
        "dry_run: False",
        "robot_agent_uid: 0",
        "human_agent_uid: 1",
        "",
        "agents:",
    ])

    for i in range(n_agents):
        lines.extend([
            f"  agent_{i}:",
            f"    uid: {i}",
        ])

    lines.extend([
        "",
        "paths:",
        "  results_dir: ${hydra:runtime.output_dir}/results",
        '  epi_result_file_path: "${paths.results_dir}/episode_result_log.csv"',
        '  run_result_file_path: "${paths.results_dir}/run_result_log.csv}"',
        '  end_result_file_path: "${paths.results_dir}/end_result_log.csv}"',
        "",
        "evaluation:",
        "  do_print: True",
        "  save_video: True",
        "  log_data: True",
        "  save_rgb: False",
        "  log_detailed_traces: True",
        "  agents:",
    ])

    for i in range(n_agents):
        lines.extend([
            f"    agent_{i}:",
            f"      uid: {i}",
        ])

    lines.extend([
        "",
        "habitat:",
        "  task:",
        "    pddl_domain_def: fp",
        "",
        "hydra:",
        "  job:",
        f"    name: 'emtom_{n_agents}_humans'",
        "    chdir: False",
        "  run:",
        "    dir: outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}-${file_stem:${habitat.dataset.data_path}}",
    ])

    return "\n".join(lines) + "\n"


def main():
    """Generate config files for 6-10 agents."""

    conf_dir = Path(__file__).parent.parent / "habitat_llm" / "conf"

    # Ensure directories exist
    (conf_dir / "habitat_conf").mkdir(exist_ok=True)
    (conf_dir / "habitat_conf" / "task").mkdir(exist_ok=True)
    (conf_dir / "evaluation").mkdir(exist_ok=True)
    (conf_dir / "examples").mkdir(exist_ok=True)

    for n in range(6, 11):  # 6 to 10 agents
        print(f"Generating configs for {n} agents...")

        # habitat_conf
        path = conf_dir / "habitat_conf" / f"emtom_spot_multi_agent_{n}.yaml"
        path.write_text(generate_habitat_conf(n))
        print(f"  Created {path.name}")

        # task
        path = conf_dir / "habitat_conf" / "task" / f"rearrange_easy_multi_agent_{n}.yaml"
        path.write_text(generate_task_conf(n))
        print(f"  Created {path.name}")

        # evaluation
        path = conf_dir / "evaluation" / f"decentralized_evaluation_runner_{n}_agent_openai.yaml"
        path.write_text(generate_evaluation_conf(n))
        print(f"  Created {path.name}")

        # examples - robots
        path = conf_dir / "examples" / f"emtom_{n}_robots.yaml"
        path.write_text(generate_example_robots_conf(n))
        print(f"  Created {path.name}")

        # examples - humans (skip for now, can add if needed)
        # path = conf_dir / "examples" / f"emtom_{n}_humans.yaml"
        # path.write_text(generate_example_humans_conf(n))
        # print(f"  Created {path.name}")

    print("\nDone! Generated config files for 6-10 agents.")
    print("Remember to also update run_emtom.sh to support 6-10 agents in get_agent_config().")


if __name__ == "__main__":
    main()
