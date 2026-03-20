from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


class ExternalAgentError(RuntimeError):
    pass


class ExternalAgentLauncher:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.base_tmp_dir = project_root / "tmp" / "task_gen"
        self.agent_env_dir = self.base_tmp_dir / ".venv"

    @property
    def agent_python(self) -> Path:
        return self.agent_env_dir / "bin" / "python"

    def ensure_agent_environment(self, agent_name: str) -> Path:
        self.base_tmp_dir.mkdir(parents=True, exist_ok=True)
        if not self.agent_python.exists():
            cmd = ["uv", "venv", str(self.agent_env_dir), "--python", sys.executable]
            self._run_bootstrap(cmd, "create task-gen agent environment")

        return self.agent_env_dir

    def _run_bootstrap(self, cmd: List[str], description: str) -> None:
        try:
            subprocess.run(cmd, cwd=str(self.project_root), check=True)
        except subprocess.CalledProcessError as exc:
            raise ExternalAgentError(f"Failed to {description}: {exc}") from exc

    def build_agent_env(
        self,
        *,
        workspace_dir: Path,
        inherit_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        base_env = dict(inherit_env or os.environ)
        for key in [
            "CONDA_DEFAULT_ENV",
            "CONDA_PREFIX",
            "CONDA_PROMPT_MODIFIER",
            "CONDA_SHLVL",
            "PYTHONHOME",
            "PYTHONPATH",
        ]:
            base_env.pop(key, None)

        env = dict(base_env)
        path_parts = [
            str(workspace_dir / "bin"),
            str(self.agent_env_dir / "bin"),
            env.get("PATH", ""),
        ]
        env["PATH"] = os.pathsep.join(part for part in path_parts if part)
        env["VIRTUAL_ENV"] = str(self.agent_env_dir)
        env["PAGER"] = "cat"
        env["MANPAGER"] = "cat"
        env["LESS"] = "-R"
        env["PIP_PROGRESS_BAR"] = "off"
        env["TQDM_DISABLE"] = "1"
        return env

    def resolve_executable(self, agent_name: str, env: Dict[str, str]) -> str:
        executable_names = {
            "mini": "mini",
            "claude": "claude",
            "codex": "codex",
        }
        executable = executable_names[agent_name]
        resolved = shutil.which(executable, path=env.get("PATH"))
        if not resolved:
            raise ExternalAgentError(
                f"Could not find executable '{executable}' for task-gen agent '{agent_name}'."
            )
        return resolved

    def build_command(
        self,
        *,
        agent_name: str,
        executable: str,
        workspace_dir: Path,
        bootstrap_prompt: str,
        model: Optional[str] = None,
    ) -> List[str]:
        if agent_name == "mini":
            cmd = [executable, "-y"]
            if model:
                cmd.extend(["-m", model])
            cmd.extend(["-t", bootstrap_prompt])
            return cmd
        if agent_name == "claude":
            cmd = [
                executable,
                "--bare",
                "--dangerously-skip-permissions",
                "--print",
                "--output-format",
                "text",
                "--add-dir",
                str(workspace_dir),
                "--tools",
                "Bash",
            ]
            if model:
                cmd.extend(["--model", model])
            cmd.append(bootstrap_prompt)
            return cmd
        if agent_name == "codex":
            cmd = [
                executable,
                "exec",
                "--sandbox",
                "workspace-write",
                "--ask-for-approval",
                "never",
                "--cd",
                str(workspace_dir),
            ]
            if model:
                cmd.extend(["--model", model])
            cmd.append(bootstrap_prompt)
            return cmd
        raise ExternalAgentError(f"Unsupported task-gen agent: {agent_name}")

    def run(
        self,
        *,
        agent_name: str,
        workspace_dir: Path,
        bootstrap_prompt: str,
        model: Optional[str] = None,
    ) -> int:
        self.ensure_agent_environment(agent_name)
        env = self.build_agent_env(workspace_dir=workspace_dir)
        executable = self.resolve_executable(agent_name, env)
        cmd = self.build_command(
            agent_name=agent_name,
            executable=executable,
            workspace_dir=workspace_dir,
            bootstrap_prompt=bootstrap_prompt,
            model=model,
        )
        proc = subprocess.run(cmd, cwd=str(workspace_dir), env=env)
        return proc.returncode
