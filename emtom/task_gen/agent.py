"""
Agentic task generator for EMTOM benchmark.

A ReAct-style agent with 3 tools:
- bash: Shell commands for file exploration and task editing
- test_task: Run benchmark with current working_task.json
- submit_task: Save current task to output directory
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig

from .prompts import SYSTEM_PROMPT

if TYPE_CHECKING:
    from habitat_llm.llm.base_llm import BaseLLM


class TaskGeneratorAgent:
    """
    ReAct-style agent for iterative task generation.

    Uses only 3 tools:
    - bash: For file exploration and task editing
    - test_task: Run benchmark with current task
    - submit_task: Save curated task to output
    """

    def __init__(
        self,
        llm_client: "BaseLLM",
        config: DictConfig,
        working_dir: str = "data/emtom/tasks",
        trajectory_dir: str = "data/emtom/trajectories",
        output_dir: str = "data/emtom/tasks/curated",
        max_iterations: int = 100,
        verbose: bool = True,
    ):
        """
        Initialize the agent.

        Args:
            llm_client: LLM client for agent reasoning
            config: Hydra config for BenchmarkRunner
            working_dir: Directory containing working_task.json
            trajectory_dir: Directory with trajectory JSON files
            output_dir: Directory for curated output tasks
            max_iterations: Max ReAct iterations before stopping
            verbose: Print agent thoughts and actions
        """
        self.llm = llm_client
        self.config = config
        self.working_dir = Path(working_dir)
        self.trajectory_dir = Path(trajectory_dir)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Task file paths
        self.task_file = self.working_dir / "working_task.json"
        self.template_file = self.working_dir / "template.json"

        # Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track state
        self.submitted_tasks: List[str] = []
        self.messages: List[Dict[str, str]] = []
        self.iteration_count = 0

        # Whitelisted bash commands (for safety)
        self.allowed_commands = [
            "ls", "cat", "head", "tail", "grep", "jq", "wc",
            "echo", "sed", "awk", "sort", "uniq", "find"
        ]

    def run(self, num_tasks_target: int = 1) -> List[str]:
        """
        Run the agent to generate tasks.

        Args:
            num_tasks_target: Number of quality tasks to generate

        Returns:
            List of paths to submitted task files
        """
        self._log(f"\n{'='*60}")
        self._log(f"Starting TaskGeneratorAgent")
        self._log(f"Target: {num_tasks_target} tasks")
        self._log(f"Trajectories: {self.trajectory_dir}")
        self._log(f"Output: {self.output_dir}")
        self._log(f"{'='*60}\n")

        # Initialize conversation
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Initial user message
        user_msg = f"""Create {num_tasks_target} quality benchmark tasks.

Available trajectories are in: {self.trajectory_dir}
Template is at: {self.template_file}
Edit this file for testing: {self.task_file}

Start by exploring the available trajectories with bash."""

        self.messages.append({"role": "user", "content": user_msg})

        # Main ReAct loop
        while len(self.submitted_tasks) < num_tasks_target:
            self.iteration_count += 1

            if self.iteration_count > self.max_iterations:
                self._log(f"\nReached max iterations ({self.max_iterations})")
                break

            self._log(f"\n--- Iteration {self.iteration_count} ---")

            # Get LLM response
            try:
                response = self._get_llm_response()
            except Exception as e:
                self._log(f"LLM error: {e}")
                self.messages.append({
                    "role": "user",
                    "content": f"Error getting LLM response: {e}. Please try again."
                })
                continue

            # Parse action from response
            action = self._parse_action(response)

            if action is None:
                self._log("No valid action found in response")
                self.messages.append({
                    "role": "assistant",
                    "content": response
                })
                self.messages.append({
                    "role": "user",
                    "content": "I couldn't parse a valid action. Please respond with:\nThought: [reasoning]\nAction: tool_name[args]"
                })
                continue

            # Execute action and get observation
            tool, args = action
            observation = self._execute_action(tool, args)

            # Add to conversation
            self.messages.append({"role": "assistant", "content": response})
            self.messages.append({"role": "user", "content": f"Observation: {observation}"})

            self._log(f"Observation: {observation[:500]}...")

        self._log(f"\n{'='*60}")
        self._log(f"Agent finished. Submitted {len(self.submitted_tasks)} tasks:")
        for task_path in self.submitted_tasks:
            self._log(f"  - {task_path}")
        self._log(f"{'='*60}\n")

        return self.submitted_tasks

    def _get_llm_response(self) -> str:
        """Get response from LLM."""
        # Format messages for LLM
        prompt = self._format_messages_for_llm()
        response = self.llm.generate(prompt)
        self._log(f"Agent: {response[:300]}...")
        return response

    def _format_messages_for_llm(self) -> str:
        """Format message history for the LLM."""
        parts = []
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System:\n{content}")
            elif role == "user":
                parts.append(f"User:\n{content}")
            elif role == "assistant":
                parts.append(f"Assistant:\n{content}")
        return "\n\n".join(parts)

    def _parse_action(self, response: str) -> Optional[tuple]:
        """
        Parse action from LLM response.

        Expected format:
        Thought: [reasoning]
        Action: tool_name[args]

        Returns:
            Tuple of (tool_name, args) or None if parsing fails
        """
        # First, try to match heredoc pattern (most complex case)
        # Matches: Action: bash[cat > file << 'EOF'\n...\nEOF]
        heredoc_pattern = r"Action:\s*bash\[(cat\s*>\s*[^\n]+<<\s*'?EOF'?\n.*?\nEOF)\]"
        match = re.search(heredoc_pattern, response, re.DOTALL)
        if match:
            return ("bash", match.group(1).strip())

        # Try matching jq -n pattern with multi-line content
        jq_pattern = r"Action:\s*bash\[(jq\s+-n\s+'[^']*'(?:\s*>\s*[^\]]+)?)\]"
        match = re.search(jq_pattern, response, re.DOTALL)
        if match:
            return ("bash", match.group(1).strip())

        # Try simple Action: tool_name[args] pattern (single line or simple args)
        # Find "Action:" then match tool name and extract args up to the closing ]
        simple_pattern = r'Action:\s*(\w+)\[([^\]]*)\]'
        match = re.search(simple_pattern, response)
        if match:
            tool = match.group(1).strip()
            args = match.group(2).strip()
            return (tool, args)

        # For test_task[] and submit_task[] with empty args
        empty_args_pattern = r'Action:\s*(\w+)\[\s*\]'
        match = re.search(empty_args_pattern, response)
        if match:
            return (match.group(1).strip(), "")

        return None

    def _execute_action(self, tool: str, args: str) -> str:
        """Execute the specified tool."""
        self._log(f"Executing: {tool}[{args[:100]}...]")

        if tool == "bash":
            return self._bash(args)
        elif tool == "test_task":
            return self._test_task()
        elif tool == "submit_task":
            return self._submit_task()
        else:
            return f"Unknown tool: {tool}. Available tools: bash, test_task, submit_task"

    def _bash(self, command: str) -> str:
        """
        Execute a shell command with safety limits.

        Allows file exploration and editing within allowed directories only.
        """
        # Block command chaining operators
        dangerous_patterns = ["&&", "||", ";", "|", "`", "$(", "${"]
        for pattern in dangerous_patterns:
            if pattern in command:
                return f"Command chaining not allowed: '{pattern}' detected. Use separate commands."

        # Allowed directories (relative to project root)
        allowed_paths = ["data/emtom/", "outputs/emtom/", "emtom/"]

        # Basic safety check - command should start with allowed command
        first_word = command.split()[0] if command.split() else ""

        # Allow cat with heredoc even though it writes
        is_heredoc = "<<" in command and "EOF" in command

        # Check if command is allowed
        is_allowed = first_word in self.allowed_commands or is_heredoc

        if not is_allowed:
            return f"Command not allowed: {first_word}. Allowed: {', '.join(self.allowed_commands)}, cat with heredoc"

        # For heredoc writes, validate the target path
        if is_heredoc and ">" in command:
            # Extract path after "cat >" or "cat>", before "<<"
            path_match = re.search(r'cat\s*>\s*([^\s<]+)', command)
            if path_match:
                target_path = path_match.group(1).strip()
                if not any(target_path.startswith(p) for p in allowed_paths):
                    return f"Write not allowed: {target_path}. Can only write to: {', '.join(allowed_paths)}"

        # For read commands, check if accessing sensitive paths
        sensitive_paths = ["/etc/", "/root/", "~/.ssh", ".env", "credentials", "secret"]
        command_lower = command.lower()
        for sensitive in sensitive_paths:
            if sensitive in command_lower:
                return f"Access denied: cannot access paths containing '{sensitive}'"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path.cwd())  # Run from project root
            )
            output = result.stdout + result.stderr
            if not output.strip():
                output = "(no output)"
            # Truncate long output
            if len(output) > 5000:
                output = output[:5000] + "\n... (truncated)"
            return output
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
        except Exception as e:
            return f"Command failed: {e}"

    def _test_task(self) -> str:
        """
        Validate and optionally run benchmark with current working_task.json.

        First validates the task structure, then attempts to run benchmark
        if the environment is available.
        """
        # Check task file exists
        if not self.task_file.exists():
            return "Error: working_task.json does not exist. Create it first with bash."

        # Load task
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON in working_task.json: {e}"

        self._log(f"Testing task: {task_data.get('title', 'Untitled')}")

        # First, validate task structure
        validation_result = self._validate_task_structure(task_data)
        if "error" in validation_result:
            return json.dumps(validation_result, indent=2)

        # Try to run benchmark if environment is available
        try:
            results = self._run_benchmark(task_data)
            # Check if benchmark returned an error (import error, etc.)
            if results.get("error"):
                # Merge validation result with benchmark error
                validation_result["benchmark_error"] = results["error"]
                validation_result["summary"] = f"Task structure valid. Benchmark skipped: {results['error']}"
                return json.dumps(validation_result, indent=2)
            # Benchmark ran successfully - merge results with validation
            validation_result.update(results)
            return json.dumps(validation_result, indent=2)
        except Exception as e:
            # If benchmark fails due to environment issues, return validation result
            # with a note that benchmark couldn't run
            validation_result["benchmark_error"] = str(e)
            validation_result["summary"] = f"Task structure valid. Benchmark skipped: {e}"
            return json.dumps(validation_result, indent=2)

    def _validate_task_structure(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task JSON structure without running benchmark."""
        required_fields = [
            "task_id", "title", "public_goal", "category",
            "mechanic_bindings", "agent_secrets", "agent_roles",
            "agent_actions", "success_condition"
        ]

        missing = [f for f in required_fields if f not in task_data]
        if missing:
            return {
                "valid": False,
                "error": f"Missing required fields: {missing}",
                "summary": "Task validation failed"
            }

        # Check mechanic_bindings structure
        for i, binding in enumerate(task_data.get("mechanic_bindings", [])):
            if "mechanic_type" not in binding:
                return {
                    "valid": False,
                    "error": f"mechanic_bindings[{i}] missing mechanic_type",
                    "summary": "Task validation failed"
                }
            if "trigger_object" not in binding:
                return {
                    "valid": False,
                    "error": f"mechanic_bindings[{i}] missing trigger_object",
                    "summary": "Task validation failed"
                }

        # Check agent_secrets has proper structure
        if not isinstance(task_data.get("agent_secrets"), dict):
            return {
                "valid": False,
                "error": "agent_secrets must be a dict",
                "summary": "Task validation failed"
            }

        # Try to parse as GeneratedTask
        try:
            from emtom.task_gen import GeneratedTask
            task = GeneratedTask.from_dict(task_data)
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to parse as GeneratedTask: {e}",
                "summary": "Task validation failed"
            }

        return {
            "valid": True,
            "task_id": task_data.get("task_id"),
            "title": task_data.get("title"),
            "mechanics": [b.get("mechanic_type") for b in task_data.get("mechanic_bindings", [])],
            "tom_required": task_data.get("theory_of_mind_required", False),
            "summary": "Task structure is valid"
        }

    def _run_benchmark(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the actual benchmark test.

        This sets up the Habitat environment, creates agents, and runs the task.
        """
        # Import here to avoid circular imports and allow running without habitat
        try:
            from emtom.runner import BenchmarkRunner
            from emtom.runner.benchmark import task_to_instruction
            from emtom.task_gen import GeneratedTask
            from habitat_llm.agent.env import register_actions, register_measures, register_sensors
            from habitat_llm.world_model.world_model import CollaborationDatasetV0
            from habitat_llm.agent.env.environment_interface import EnvironmentInterface
        except ImportError as e:
            return {
                "steps": 0,
                "done": False,
                "episode_over": True,
                "error": f"Import error: {e}",
                "summary": "Could not import benchmark dependencies"
            }

        # Convert task_data to GeneratedTask
        try:
            task = GeneratedTask.from_dict(task_data)
        except Exception as e:
            return {
                "steps": 0,
                "done": False,
                "episode_over": True,
                "error": f"Invalid task format: {e}",
                "summary": "Task JSON is malformed"
            }

        # Setup environment
        output_dir = f"outputs/emtom/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Register components
            register_sensors(self.config)
            register_actions(self.config)
            register_measures(self.config)

            # Create environment
            dataset = CollaborationDatasetV0(self.config.habitat.dataset)
            env_interface = EnvironmentInterface(self.config, dataset=dataset, init_wg=False)
            env_interface.initialize_perception_and_world_graph()

            # Create runner
            runner = BenchmarkRunner(self.config)

            # Convert task to mechanics dict for initialization
            task_mechanics = {
                "mechanics": [
                    {"mechanic_type": b.mechanic_type, **b.to_dict()}
                    for b in task.mechanic_bindings
                ]
            } if task.mechanic_bindings else None

            runner.setup(
                env_interface=env_interface,
                task_data=task_mechanics,
                output_dir=output_dir,
                task=task,
            )

            # Generate instruction
            instruction = task_to_instruction(task)

            # Run benchmark (limited steps for testing)
            results = runner.run(
                instruction=instruction,
                max_steps=100,  # Cap at 100 for quick testing
            )

            runner.cleanup()

            return {
                "steps": results.get("steps", 0),
                "done": results.get("done", False),
                "episode_over": results.get("episode_over", False),
                "summary": f"Task {'completed' if results.get('done') else 'not completed'} in {results.get('steps', 0)} steps"
            }

        except Exception as e:
            return {
                "steps": 0,
                "done": False,
                "episode_over": True,
                "error": str(e),
                "summary": f"Benchmark error: {e}"
            }

    def _submit_task(self) -> str:
        """
        Copy working task to output directory.

        Called when the agent determines task quality is good.
        """
        if not self.task_file.exists():
            return "Error: working_task.json does not exist. Create and test it first."

        # Load task to validate
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON: {e}"

        # Generate task ID
        task_id = task_data.get("task_id", f"task_{len(self.submitted_tasks):03d}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{task_id}_{timestamp}.json"
        output_path = self.output_dir / output_filename

        # Copy file
        shutil.copy(self.task_file, output_path)
        self.submitted_tasks.append(str(output_path))

        return f"Task saved to {output_path}. Total submitted: {len(self.submitted_tasks)}"

    def _log(self, message: str) -> None:
        """Print log message if verbose mode is on."""
        if self.verbose:
            print(message)
