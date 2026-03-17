from omegaconf import OmegaConf

from emtom.actions.baseline_tools import ReadAgentTrajectoryTool
from emtom.evolve.benchmark_wrapper import find_calibration_entry
from emtom.runner.benchmark import BenchmarkRunner, task_to_instruction


class _Task:
    def __init__(self):
        self.num_agents = 2
        self.task = "Move the object to the target."
        self.agent_actions = {
            "agent_0": ["Navigate", "Communicate"],
            "agent_1": ["Navigate", "Communicate"],
        }
        self.agent_secrets = {
            "agent_0": ["the key is in the drawer"],
            "agent_1": ["the cabinet is trapped"],
        }
        self.active_mechanics = []
        self.teams = None


def test_task_to_instruction_baseline_shares_all_secrets():
    task = _Task()

    instructions = task_to_instruction(task, run_mode="baseline")

    agent_0 = instructions["agent_0"]
    agent_1 = instructions["agent_1"]

    for instruction in (agent_0, agent_1):
        assert "Baseline Mode" in instruction
        assert "agent_0 knows: the key is in the drawer" in instruction
        assert "agent_1 knows: the cabinet is trapped" in instruction
        assert "ReadAgentTrajectoryTool" in instruction


def test_resolve_agent_actions_adds_baseline_tool():
    runner = BenchmarkRunner(OmegaConf.create({"benchmark_run_mode": "baseline"}))
    task = _Task()

    actions = runner._resolve_agent_actions(task)

    assert "ReadAgentTrajectoryTool" in actions["agent_0"]
    assert "ReadAgentTrajectoryTool" in actions["agent_1"]


def test_read_agent_trajectory_tool_exposes_only_thought_and_action():
    tool = ReadAgentTrajectoryTool(agent_uid=0)
    tool.set_trajectory_store(
        [
            {
                "turn": 1,
                "agent_id": "agent_1",
                "thought": "Thought: I should open the drawer.",
                "action": "Open[drawer_1]",
                "observation": "hidden detail that should not be returned",
            }
        ]
    )

    _, response = tool.process_high_level_action("agent_1", observations={})

    assert "Thought: I should open the drawer." in response
    assert "Action: Open[drawer_1]" in response
    assert "hidden detail" not in response


def test_read_agent_trajectory_tool_state_description_does_not_require_env():
    tool = ReadAgentTrajectoryTool(agent_uid=0)
    assert tool.get_state_description() == "Reviewing teammate trajectory"


def test_find_calibration_entry_defaults_to_standard_run_mode():
    calibration = [
        {"run_mode": "standard", "agent_models": {"agent_0": "gpt-5"}},
        {"run_mode": "baseline", "agent_models": {"agent_0": "gpt-5"}},
    ]

    standard = find_calibration_entry(
        calibration,
        agent_models={"agent_0": "gpt-5"},
    )
    baseline = find_calibration_entry(
        calibration,
        agent_models={"agent_0": "gpt-5"},
        run_mode="baseline",
    )

    assert standard["run_mode"] == "standard"
    assert baseline["run_mode"] == "baseline"
