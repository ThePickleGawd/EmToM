import json
from pathlib import Path

from emtom.task_gen.external_agent import ExternalAgentLauncher
from emtom.task_gen.prompts import build_external_taskgen_prompt
from emtom.task_gen.session import TaskGenSession, default_state


def test_external_agent_launcher_builds_backend_commands(tmp_path):
    launcher = ExternalAgentLauncher(tmp_path)

    mini_cmd = launcher.build_command(
        agent_name="mini",
        executable="/tmp/mini",
        workspace_dir=tmp_path,
        bootstrap_prompt="read the prompt",
        model="gpt-5",
    )
    claude_cmd = launcher.build_command(
        agent_name="claude",
        executable="/tmp/claude",
        workspace_dir=tmp_path,
        bootstrap_prompt="read the prompt",
        model="sonnet",
    )
    codex_cmd = launcher.build_command(
        agent_name="codex",
        executable="/tmp/codex",
        workspace_dir=tmp_path,
        bootstrap_prompt="read the prompt",
        model="o3",
    )

    assert mini_cmd == ["/tmp/mini", "-y", "-m", "gpt-5", "-t", "read the prompt"]
    assert "--model" in claude_cmd and "sonnet" in claude_cmd
    assert claude_cmd[-1] == "read the prompt"
    assert codex_cmd[:5] == ["/tmp/codex", "exec", "--sandbox", "workspace-write", "--ask-for-approval"]
    assert "--model" in codex_cmd and "o3" in codex_cmd


def test_build_agent_env_clears_conda_and_prefers_workspace(tmp_path):
    launcher = ExternalAgentLauncher(tmp_path)
    launcher.agent_env_dir.mkdir(parents=True)
    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "bin").mkdir(parents=True)

    env = launcher.build_agent_env(
        workspace_dir=workspace_dir,
        inherit_env={
            "PATH": "/usr/bin",
            "CONDA_PREFIX": "/conda",
            "CONDA_DEFAULT_ENV": "habitat-llm",
        },
    )

    assert "CONDA_PREFIX" not in env
    assert "CONDA_DEFAULT_ENV" not in env
    assert env["PATH"].split(":")[0] == str(workspace_dir / "bin")
    assert env["VIRTUAL_ENV"] == str(launcher.agent_env_dir)


def test_ensure_agent_environment_only_creates_sandbox_env(monkeypatch, tmp_path):
    launcher = ExternalAgentLauncher(tmp_path)
    calls = []

    def fake_run_bootstrap(cmd, description):
        calls.append((cmd, description))
        launcher.agent_env_dir.mkdir(parents=True, exist_ok=True)
        (launcher.agent_env_dir / "bin").mkdir(exist_ok=True)
        (launcher.agent_env_dir / "bin" / "python").write_text("")

    monkeypatch.setattr(launcher, "_run_bootstrap", fake_run_bootstrap)
    launcher.ensure_agent_environment("mini")

    assert len(calls) == 1
    assert calls[0][1] == "create task-gen agent environment"


def test_build_external_prompt_rewrites_taskgen_commands():
    prompt = build_external_taskgen_prompt(
        working_dir="/repo/tmp/task_gen/run",
        task_file="/repo/tmp/task_gen/run/working_task.json",
        category="cooperative",
        available_items="item_a",
        available_mechanics="room_restriction",
        available_predicates="is_open",
        action_descriptions="Navigate",
        extra_sections="## Required K-Level: 2",
        num_tasks=1,
        agents_min=2,
        agents_max=3,
        subtasks_min=2,
        subtasks_max=4,
    )

    assert "taskgen new_scene N" in prompt
    assert "taskgen judge" in prompt
    assert "taskgen submit_task" in prompt
    assert "taskgen finish" in prompt
    assert "judge[]" not in prompt
    assert "submit_task[]" not in prompt


def test_taskgen_session_finish_and_fail(tmp_path):
    state = default_state(
        working_dir=str(tmp_path),
        output_dir="data/emtom/tasks",
        num_tasks_target=2,
        agents_min=2,
        agents_max=2,
        subtasks_min=2,
        subtasks_max=4,
        category=None,
        seed_task=None,
        seed_tasks_dir=None,
        random_seed_task=False,
        judge_threshold=None,
        difficulty=None,
        test_model=None,
        calibration_stats={"model": "gpt-5.2", "target_rate": 0.20},
        task_gen_agent="mini",
        allowed_k_levels=[2],
    )
    with open(tmp_path / "taskgen_state.json", "w") as f:
        json.dump(state, f, indent=2)

    session = TaskGenSession(str(tmp_path))
    incomplete = session.finish()
    assert incomplete["success"] is False

    session.state["submitted_tasks"] = ["a.json", "b.json"]
    session._write_state()
    complete = session.finish()
    assert complete["success"] is True

    failed = session.fail("boom")
    assert failed["success"] is True

    saved = json.loads((tmp_path / "taskgen_state.json").read_text())
    assert saved["finished"] is True
    assert saved["failed"] is True
    assert saved["fail_reason"] == "boom"
