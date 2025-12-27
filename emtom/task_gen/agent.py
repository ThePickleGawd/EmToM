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
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from omegaconf import DictConfig

from .prompts import SYSTEM_PROMPT
from emtom.actions import ActionRegistry

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
        output_dir: str = "data/emtom/tasks",
        max_iterations: int = 100,
        verbose: bool = True,
        subtasks: int = 3,
        scene_data: Optional[Any] = None,
        log_dir: Optional[str] = None,
        max_context_chars: Optional[int] = None,
    ):
        """
        Initialize the agent.

        Args:
            llm_client: LLM client for agent reasoning
            config: Hydra config for BenchmarkRunner
            working_dir: Directory containing working_task.json
            output_dir: Directory for curated output tasks
            max_iterations: Max ReAct iterations before stopping
            verbose: Print agent thoughts and actions
            subtasks: Exact number of subtasks/steps per task
            scene_data: Live scene data from SceneLoader (required)
            log_dir: Directory for log files (defaults to Hydra output or output_dir/logs)
            max_context_chars: Max context size before summarizing. Auto-detected from model if None.
        """
        self.llm = llm_client
        self.config = config
        self.working_dir = Path(working_dir)
        self.subtasks = subtasks
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.scene_data = scene_data
        self.max_context_chars = max_context_chars or self._get_model_context_limit()

        # Task file paths
        self.task_file = self.working_dir / "working_task.json"
        self.template_file = self.working_dir / "template.json"

        # Create directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Copy fresh template from source to working directory
        source_template = Path(__file__).parent / "template" / "template.json"
        if source_template.exists():
            shutil.copy(source_template, self.template_file)

        # Track state
        self.submitted_tasks: List[str] = []
        self.messages: List[Dict[str, str]] = []
        self.iteration_count = 0
        self.last_verify_passed = False  # Track if golden trajectory verified
        self.failed = False  # Track if agent called fail[]
        self.fail_reason = ""  # Reason for failure

        # Setup logging to file
        # Prefer log_dir (Hydra output), fallback to output_dir/logs
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"task_generator_{timestamp}.log"
        self._log_handle = open(self.log_file, "w")

        # Whitelisted bash commands (for safety)
        self.allowed_commands = [
            # File operations
            "ls", "cat", "head", "tail", "grep", "jq", "wc",
            "echo", "sed", "awk", "sort", "uniq", "find",
            # Shell control structures (for loops, conditionals)
            "for", "do", "done", "while", "if", "then", "else", "fi", "in",
        ]

    def _get_model_context_limit(self) -> int:
        """
        Get context limit based on model name.

        Returns 80% of model's context window in chars (~4 chars/token).
        """
        # Model context windows (in tokens)
        context_windows = {
            "gpt-5.2": 128000,
            "gpt-5": 128000,
            "gpt-5-mini": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "claude-3": 200000,
            "claude-3.5": 200000,
        }

        # Try to get model name from LLM config
        model_name = ""
        try:
            if hasattr(self.llm, 'llm_conf'):
                params = self.llm.llm_conf.get('generation_params', {})
                model_name = params.get('model', '').lower()
        except Exception:
            pass

        # Find matching context window
        for model_key, tokens in context_windows.items():
            if model_key in model_name:
                # 80% of context, ~4 chars per token
                return int(tokens * 0.8 * 4)

        # Default: assume 128k context (gpt-5 class)
        return int(128000 * 0.8 * 4)

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
        self._log(f"Output: {self.output_dir}")
        self._log(f"{'='*60}\n")

        if not self.scene_data:
            self._log("ERROR: No scene_data provided!")
            return []

        self._log(f"Scene: {self.scene_data.scene_id} (episode {self.scene_data.episode_id})")
        self._log(f"  Rooms: {len(self.scene_data.rooms)}")
        self._log(f"  Furniture: {len(self.scene_data.furniture)}")
        self._log(f"  Objects: {len(self.scene_data.objects)}")

        # Clean up working_task.json from previous runs
        if self.task_file.exists():
            self.task_file.unlink()
            self._log(f"Cleaned up previous {self.task_file}")

        # Create working_task.json from template with scene fields pre-populated
        self._create_working_task_from_template()

        # Load scenario inspirations - one per task target
        # Truncate long scenarios to avoid confusing the agent
        from emtom.task_gen.task_generator import load_scenario_inspirations
        scenarios = load_scenario_inspirations(max_scenarios=num_tasks_target)
        scenario_text = ""
        if scenarios:
            scenario_text = "\n\n## Story Inspiration (for atmosphere only - do NOT copy!)\n"
            scenario_text += "Use these themes as creative inspiration for your story. Write your OWN story using the REAL object IDs from the scene.\n"
            for i, s in enumerate(scenarios, 1):
                # Truncate to ~500 chars to avoid overwhelming context
                truncated = s[:500] + "..." if len(s) > 500 else s
                scenario_text += f"\n--- Theme {i} ---\n{truncated}\n"

        # Get available items from registry
        from emtom.state.item_registry import ItemRegistry
        available_items = ItemRegistry.get_items_for_task_generation()

        # Get available mechanics
        from emtom.mechanics import get_mechanics_for_task_generation
        available_mechanics = get_mechanics_for_task_generation()

        # Initialize conversation with action descriptions and paths injected
        system_prompt = SYSTEM_PROMPT.replace(
            "{action_descriptions}",
            ActionRegistry.get_all_action_descriptions()
        ).replace(
            "{template_file}",
            str(self.template_file)
        ).replace(
            "{task_file}",
            str(self.task_file)
        ).replace(
            "{output_dir}",
            str(self.output_dir)
        ).replace(
            "{available_items}",
            available_items
        ).replace(
            "{available_mechanics}",
            available_mechanics
        )
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Format scene data for the prompt
        scene_info = self._format_scene_data()

        # Initial user message with scene data
        user_msg = f"""Create {num_tasks_target} quality benchmark tasks.

## Task Requirements
- **Subtasks**: Exactly {self.subtasks} steps per task
- Each subtask should be a distinct action that gates progress to the next

## Scene Data (from PARTNR dataset - these objects EXIST!)
**Episode ID**: {self.scene_data.episode_id}
**Scene ID**: {self.scene_data.scene_id}

NOTE: scene_id and episode_id are ALREADY SET in working_task.json. Do not modify them.

{scene_info}

Template is at: {self.template_file}
Working task (pre-populated with scene fields): {self.task_file}
{scenario_text}
IMPORTANT:
- Use ONLY the objects listed above - they are verified to exist in this scene
- The story should be atmospheric and reference REAL object IDs from the scene
- Choose mechanics to apply (inverse_state, remote_control, etc.) based on the articulated furniture available

Start by reading the template, then create a task using the scene data above."""

        self.messages.append({"role": "user", "content": user_msg})

        # Save initial context window
        self._save_context_window()

        # Main ReAct loop
        while len(self.submitted_tasks) < num_tasks_target and not self.failed:
            self.iteration_count += 1

            if self.iteration_count > self.max_iterations:
                self._log(f"\nReached max iterations ({self.max_iterations})")
                break

            self._log(f"\n{'='*40}")
            self._log(f"Iteration {self.iteration_count}/{self.max_iterations} | Submitted: {len(self.submitted_tasks)}/{num_tasks_target}")
            self._log(f"{'='*40}")

            # Check if context needs summarization
            self._maybe_summarize_context()

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

            # Truncate heredocs in PREVIOUS assistant messages to save context
            # (keep current response full so LLM sees what it just did)
            self._truncate_old_heredocs()

            # Add current turn to conversation (full content)
            self.messages.append({"role": "assistant", "content": response})
            self.messages.append({"role": "user", "content": f"Observation: {observation}"})

            self._log(f"Observation: {observation}", truncate_terminal=300)

            # Save full context window after each step
            self._save_context_window()

        self._log(f"\n{'='*60}")
        if self.failed:
            self._log(f"Agent FAILED: {self.fail_reason}")
        else:
            self._log(f"Agent finished. Submitted {len(self.submitted_tasks)} tasks:")
            for task_path in self.submitted_tasks:
                self._log(f"  - {task_path}")
        self._log(f"{'='*60}\n")

        # Save working_task.json to log directory for debugging
        self._save_working_task_to_logs()

        return self.submitted_tasks

    def _save_working_task_to_logs(self) -> None:
        """Save a copy of working_task.json to log directory for debugging."""
        if self.task_file.exists():
            try:
                import shutil
                dest = self.log_dir / "working_task_final.json"
                shutil.copy(self.task_file, dest)
                self._log(f"Saved working task to: {dest}")
            except Exception as e:
                self._log(f"Warning: Could not save working task: {e}")

    def _save_context_window(self) -> None:
        """Save the entire context window to prompt.txt for debugging."""
        prompt_file = self.log_dir / "prompt.txt"
        try:
            with open(prompt_file, "w") as f:
                for i, msg in enumerate(self.messages):
                    role = msg["role"].upper()
                    content = msg["content"]
                    f.write(f"=== {role} [{i}] ===\n\n")
                    f.write(content)
                    f.write("\n\n")
        except Exception as e:
            self._log(f"Warning: Could not save context window: {e}")

    def _create_working_task_from_template(self) -> None:
        """Create working_task.json from template with scene fields pre-populated."""
        # Load template
        with open(self.template_file) as f:
            task = json.load(f)

        # Auto-populate scene fields
        if self.scene_data:
            task["scene_id"] = self.scene_data.scene_id
            task["episode_id"] = self.scene_data.episode_id

        # Write to working_task.json
        with open(self.task_file, 'w') as f:
            json.dump(task, f, indent=2)

        self._log(f"Created {self.task_file} with scene fields pre-populated")

    def _fix_scene_fields(self) -> None:
        """Auto-fix scene fields after any write to working_task.json.

        The LLM sometimes overwrites these fields with wrong values/formats.
        This ensures they always have the correct values from the loaded scene.
        """
        if not self.scene_data or not self.task_file.exists():
            return

        try:
            with open(self.task_file) as f:
                task = json.load(f)

            # Fix the scene fields to match loaded scene
            changed = False
            if task.get("scene_id") != self.scene_data.scene_id:
                task["scene_id"] = self.scene_data.scene_id
                changed = True
            if task.get("episode_id") != self.scene_data.episode_id:
                task["episode_id"] = self.scene_data.episode_id
                changed = True

            if changed:
                with open(self.task_file, 'w') as f:
                    json.dump(task, f, indent=2)
                self._log(f"Auto-fixed scene fields in {self.task_file.name}")
        except Exception as e:
            self._log(f"Warning: Could not fix scene fields: {e}")

    def _truncate_old_heredocs(self) -> None:
        """Truncate large heredoc content in PREVIOUS assistant messages.

        When the LLM writes a file via heredoc, the entire file content
        appears in the response. For older messages, we truncate to save context.
        The most recent messages are kept full.
        """
        # Pattern: cat > file << 'EOF' ... EOF (or << "EOF" or << EOF)
        heredoc_pattern = re.compile(
            r"(cat\s*>\s*[^\s]+\s*<<\s*['\"]?EOF['\"]?\n)"  # Opening
            r"(.*?)"  # Content (non-greedy)
            r"(\nEOF)",  # Closing
            re.DOTALL
        )

        def truncate_match(match):
            opening = match.group(1)
            content = match.group(2)
            closing = match.group(3)

            # Count lines and chars
            lines = content.count('\n') + 1
            chars = len(content)

            if chars > 500:  # Only truncate if content is large
                # Show first 200 chars and summary
                preview = content[:200].rsplit('\n', 1)[0]  # Clean line break
                return f"{opening}{preview}\n... [truncated: {lines} lines, {chars} chars]{closing}"
            return match.group(0)

        # Truncate all assistant messages except the last 2 (most recent turn)
        for i, msg in enumerate(self.messages[:-2] if len(self.messages) > 2 else []):
            if msg["role"] == "assistant":
                original = msg["content"]
                truncated = heredoc_pattern.sub(truncate_match, original)
                if truncated != original:
                    self.messages[i]["content"] = truncated

    def _get_llm_response(self) -> str:
        """Get response from LLM."""
        # Format messages for LLM
        prompt = self._format_messages_for_llm()
        response = self.llm.generate(prompt)
        self._log(f"Agent: {response}", truncate_terminal=300)
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

    def _estimate_context_size(self) -> int:
        """Estimate current context size in characters."""
        return sum(len(msg["content"]) for msg in self.messages)

    def _maybe_summarize_context(self) -> None:
        """
        Summarize older messages if context exceeds threshold.

        Uses the LLM to summarize, preserving intent and important observations.

        Keeps:
        - messages[0]: System prompt (exactly as-is)
        - messages[1]: Initial user message with scene data
        - Last N messages (recent conversation)

        Summarizes middle messages via LLM into a single summary.
        """
        context_size = self._estimate_context_size()
        if context_size < self.max_context_chars:
            return

        self._log(f"Context size ({context_size} chars) exceeds threshold ({self.max_context_chars}). Summarizing...")

        # Keep system prompt (0), initial user msg (1), and last 10 messages
        keep_recent = 10
        if len(self.messages) <= 2 + keep_recent:
            # Not enough messages to summarize
            return

        # Messages to summarize: from index 2 to -(keep_recent)
        to_summarize = self.messages[2:-keep_recent]
        if not to_summarize:
            return

        # Format messages for summarization
        conversation_text = []
        for msg in to_summarize:
            role = msg["role"].capitalize()
            content = msg["content"]
            # Truncate very long observations for the summary request
            if len(content) > 500:
                content = content[:500] + "..."
            conversation_text.append(f"{role}: {content}")

        summarize_prompt = f"""Summarize this conversation history for context continuity.
Preserve:
1. Your current intent/goal and progress toward it
2. Important observations (errors, successful actions, state changes)
3. Key decisions made and why
4. Current state of working_task.json (what fields have been set)

Be concise but complete. This summary replaces the conversation history.

CONVERSATION TO SUMMARIZE:
{chr(10).join(conversation_text)}

SUMMARY:"""

        # Use LLM to generate summary
        try:
            summary = self.llm.generate(summarize_prompt)
            summary = f"[CONTEXT SUMMARY - {len(to_summarize)} messages summarized by LLM]\n\n{summary}\n\n[End of summary - recent conversation follows]"
        except Exception as e:
            self._log(f"LLM summarization failed: {e}, using fallback")
            # Fallback to simple list of actions
            actions = []
            for msg in to_summarize:
                if msg["role"] == "assistant":
                    match = re.search(r'Action:\s*(\w+)\[', msg["content"])
                    if match:
                        actions.append(match.group(1))
            summary = f"[CONTEXT SUMMARY - {len(to_summarize)} messages]\nActions taken: {', '.join(actions[-30:])}\n[End of summary]"

        # Rebuild messages: system + initial + summary + recent
        self.messages = (
            self.messages[:2] +  # System + initial user
            [{"role": "user", "content": summary}] +  # Summary
            self.messages[-keep_recent:]  # Recent messages
        )

        new_size = self._estimate_context_size()
        self._log(f"Context reduced from {context_size} to {new_size} chars")

    def _parse_action(self, response: str) -> Optional[tuple]:
        """
        Parse action from LLM response using bracket-matching.

        Expected format:
        Thought: [reasoning]
        Action: tool_name[args]

        Uses proper bracket matching to handle nested brackets in JSON, etc.

        Returns:
            Tuple of (tool_name, args) or None if parsing fails
        """
        # Find "Action:" followed by tool name and opening bracket
        action_match = re.search(r'Action:\s*(\w+)\[', response)
        if not action_match:
            return None

        tool_name = action_match.group(1)
        start_idx = action_match.end()  # Position right after the opening [

        # Use bracket matching to find the closing ]
        args = self._extract_bracket_content(response, start_idx)
        if args is None:
            return None

        return (tool_name, args.strip())

    def _extract_bracket_content(self, text: str, start: int) -> Optional[str]:
        """
        Extract content between brackets using proper bracket matching.

        Args:
            text: The full text
            start: Index right after the opening bracket

        Returns:
            The content between brackets, or None if no matching bracket found
        """
        depth = 1  # We've already seen one [
        in_single_quote = False
        in_double_quote = False
        escape_next = False
        i = start

        while i < len(text) and depth > 0:
            char = text[i]

            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                i += 1
                continue

            # Track quote state (only toggle if not in the other quote type)
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            # Count brackets only when not inside quotes
            elif not in_single_quote and not in_double_quote:
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1

            i += 1

        if depth == 0:
            # i is now one past the closing bracket
            return text[start:i - 1]

        return None

    def _split_by_operators(self, command: str) -> List[str]:
        """
        Split command by shell operators (&&, ||, ;, |) while respecting quotes.

        Returns list of sub-commands.
        """
        sub_commands = []
        current = []
        in_single_quote = False
        in_double_quote = False
        i = 0

        while i < len(command):
            char = command[i]

            # Track quote state
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                current.append(char)
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                current.append(char)
            # Check for operators outside quotes
            elif not in_single_quote and not in_double_quote:
                # Check for && or ||
                if i + 1 < len(command) and command[i:i+2] in ("&&", "||"):
                    sub_commands.append("".join(current))
                    current = []
                    i += 2
                    continue
                # Check for ; or |
                elif char in ";|":
                    sub_commands.append("".join(current))
                    current = []
                else:
                    current.append(char)
            else:
                current.append(char)

            i += 1

        # Add the last command
        if current:
            sub_commands.append("".join(current))

        return sub_commands

    def _execute_action(self, tool: str, args: str) -> str:
        """Execute the specified tool."""
        self._log(f"Executing: {tool}[{args[:100]}...]")

        if tool == "bash":
            # Reset verification on any bash edit to task file
            if "working_task.json" in args and (">" in args or "cat" in args):
                self.last_verify_passed = False
            return self._bash(args)
        elif tool == "test_task":
            return self._test_task()
        elif tool == "verify_golden_trajectory":
            return self._verify_golden_trajectory()
        elif tool == "submit_task":
            return self._submit_task()
        elif tool == "fail":
            return self._fail(args)
        else:
            return f"Unknown tool: {tool}. Available: bash, test_task, verify_golden_trajectory, submit_task, fail"

    def _bash(self, command: str) -> str:
        """
        Execute a shell command with safety limits.

        Allows file exploration and editing within allowed directories only.
        Allows command chaining (&&, ||, ;, |) but validates each sub-command.
        """
        # Block command substitution (dangerous - allows arbitrary code execution)
        dangerous_patterns = ["`", "$(", "${"]
        for pattern in dangerous_patterns:
            if pattern in command:
                return f"Command substitution not allowed: '{pattern}' detected."

        # Allowed directories (temp working dir only - all working files are there)
        allowed_paths = [str(self.working_dir)]

        # Allow cat with heredoc even though it writes
        is_heredoc = "<<" in command and "EOF" in command

        # Split command by chain operators to validate each part
        # For heredocs, only validate the command part before <<
        if is_heredoc:
            command_part = command.split("<<")[0]
            heredoc_part = "<<" + command.split("<<", 1)[1]
        else:
            command_part = command
            heredoc_part = ""

        # Split by chain operators (but not inside quotes)
        sub_commands = self._split_by_operators(command_part)

        # Validate each sub-command starts with an allowed command
        for sub_cmd in sub_commands:
            sub_cmd = sub_cmd.strip()
            if not sub_cmd:
                continue
            first_word = sub_cmd.split()[0] if sub_cmd.split() else ""
            if first_word not in self.allowed_commands:
                return f"Command not allowed: '{first_word}'. Allowed: {', '.join(self.allowed_commands)}"

        # For heredoc writes, validate the target path
        if is_heredoc and ">" in command:
            # Extract path after "cat >" or "cat>", before "<<"
            path_match = re.search(r'cat\s*>\s*([^\s<]+)', command)
            if path_match:
                target_path = path_match.group(1).strip()
                if not any(target_path.startswith(p) for p in allowed_paths):
                    return f"Write not allowed: {target_path}. Can only write to: {', '.join(allowed_paths)}"

        # For read commands, check if accessing sensitive file paths
        # Only check the command part, not heredoc content
        sensitive_paths = ["/etc/", "/root/", "~/.ssh/", ".env", "credentials.json"]
        command_to_check = command_part.lower()
        for sensitive in sensitive_paths:
            if sensitive in command_to_check:
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
            # Log full output to file, but truncate for LLM context
            if len(output) > 5000:
                self._log(f"[Full bash output ({len(output)} chars)]:\n{output}")
                output = output[:5000] + f"\n... (truncated, full output in log file)"

            # Auto-fix scene fields if working_task.json was modified
            if "working_task.json" in command and (">" in command or "<<" in command):
                self._fix_scene_fields()

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

            # Log detailed action history
            action_history = results.get("action_history", [])
            if action_history:
                self._log("\n=== Agent Action History ===")
                for entry in action_history:
                    self._log(f"  Turn {entry.get('turn', '?')}: {entry.get('agent', '?')} -> {entry.get('action', '?')}")

            # Save planner traces to separate file for reference
            planner_traces = results.get("planner_traces", {})
            if planner_traces:
                trace_file = self.log_dir / f"planner_traces_{datetime.now().strftime('%H%M%S')}.txt"
                with open(trace_file, 'w') as f:
                    for agent_id, trace in planner_traces.items():
                        f.write(f"\n{'='*60}\n")
                        f.write(f"=== {agent_id} Trace ===\n")
                        f.write(f"{'='*60}\n\n")
                        f.write(trace)
                        f.write("\n")
                self._log(f"Planner traces saved to: {trace_file}")
                # Don't include full traces in the JSON response (too large)
                results["planner_traces"] = f"See {trace_file}"

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
        # Core required fields (success_condition is optional if subtasks have valid DAG)
        required_fields = [
            "task_id", "title", "story", "episode_id",
            "public_goal", "category", "mechanic_bindings", "agent_secrets",
            "agent_roles", "agent_actions"
        ]

        missing = [f for f in required_fields if f not in task_data]
        if missing:
            return {
                "valid": False,
                "error": f"Missing required fields: {missing}",
                "summary": "Task validation failed"
            }

        # Validate episode_id matches the loaded scene
        if self.scene_data:
            expected_episode = self.scene_data.episode_id
            task_episode = task_data.get("episode_id", "")
            if task_episode != expected_episode:
                return {
                    "valid": False,
                    "error": f"episode_id must be '{expected_episode}' (from loaded scene), got '{task_episode}'",
                    "summary": "Task validation failed - wrong episode"
                }

        # Collect defined item IDs from task
        defined_items = set()
        for item in task_data.get("items", []):
            item_id = item.get("item_id")
            if item_id:
                defined_items.add(item_id)

        # Validate object IDs in golden_trajectory exist in scene or are custom items
        if self.scene_data and task_data.get("golden_trajectory"):
            all_valid_ids = set(
                self.scene_data.rooms +
                self.scene_data.furniture +
                self.scene_data.objects
            )
            # Also include custom items from task definition
            all_valid_ids.update(defined_items)

            invalid_ids = []
            invalid_items = []
            for step in task_data["golden_trajectory"]:
                actions = step.get("actions", [])
                for action_entry in actions:
                    # Parse PARTNR-style action: "Navigate[table_22]" -> args = "table_22"
                    action_str = action_entry.get("action", "")
                    import re
                    match = re.match(r'(\w+)(?:\[(.+)\])?$', action_str)
                    if not match:
                        continue
                    action_name, args = match.group(1), match.group(2)

                    # Skip actions without args or communication/wait
                    if not args or action_name in ["Communicate", "Wait"]:
                        continue

                    # For Place/UseItem, args is comma-separated - check all parts
                    # Place format: Place[obj, relation, target, constraint, ref]
                    # Skip relation/constraint keywords (from habitat_llm/world_model/entities/furniture.py)
                    place_keywords = {"on", "within", "next_to"}
                    parts = [p.strip() for p in args.split(",")]
                    for target_id in parts:
                        if not target_id or target_id == "None" or target_id in place_keywords:
                            continue
                        # Check item_ prefixed IDs against defined items
                        if target_id.startswith("item_"):
                            if target_id not in defined_items:
                                invalid_items.append(target_id)
                        elif target_id not in all_valid_ids:
                            invalid_ids.append(target_id)

            if invalid_ids:
                return {
                    "valid": False,
                    "error": f"golden_trajectory contains invalid object IDs not in scene: {list(set(invalid_ids))}",
                    "summary": "Task validation failed - invalid object IDs"
                }
            if invalid_items:
                return {
                    "valid": False,
                    "error": f"golden_trajectory references undefined items: {list(set(invalid_items))}. Defined items: {list(defined_items)}",
                    "summary": "Task validation failed - item ID mismatch"
                }

        # Validate success condition OR subtasks DAG
        has_success_condition = "success_condition" in task_data and task_data["success_condition"]
        has_subtasks = "subtasks" in task_data and task_data["subtasks"]

        if not has_success_condition and not has_subtasks:
            return {
                "valid": False,
                "error": "Task must have either 'success_condition' or 'subtasks' with valid DAG",
                "summary": "Task validation failed"
            }

        # If using subtasks, validate DAG structure
        if has_subtasks:
            from emtom.task_gen.dag import validate_dag
            from emtom.task_gen import Subtask

            subtasks = []
            for s in task_data["subtasks"]:
                if isinstance(s, dict):
                    subtasks.append(Subtask.from_dict(s))

            is_valid, errors = validate_dag(subtasks)
            if not is_valid:
                return {
                    "valid": False,
                    "error": f"Invalid subtask DAG: {'; '.join(errors)}",
                    "summary": "Task validation failed"
                }

            # Validate item IDs in subtask success conditions
            invalid_items_in_subtasks = []
            for s in task_data["subtasks"]:
                if isinstance(s, dict):
                    condition = s.get("success_condition", {})
                    if isinstance(condition, dict):
                        # Check has_item conditions
                        if condition.get("property") == "has_item":
                            target_item = condition.get("target") or condition.get("value")
                            if target_item and target_item.startswith("item_"):
                                if target_item not in defined_items:
                                    invalid_items_in_subtasks.append(target_item)

            if invalid_items_in_subtasks:
                return {
                    "valid": False,
                    "error": f"subtasks reference undefined items: {list(set(invalid_items_in_subtasks))}. Defined items: {list(defined_items)}",
                    "summary": "Task validation failed - item ID mismatch in subtasks"
                }

        # Check story is not empty
        if not task_data.get("story") or len(task_data.get("story", "")) < 20:
            return {
                "valid": False,
                "error": "story field must be at least 20 characters of atmospheric narrative",
                "summary": "Task validation failed"
            }

        # Check story is grounded in real objects (must contain at least one object ID pattern)
        story = task_data.get("story", "")
        # Look for patterns like "table_59", "chest_of_drawers_54", "kettle_3"
        import re
        object_pattern = r'\b[a-z_]+_\d+\b'
        object_refs = re.findall(object_pattern, story)
        if not object_refs:
            return {
                "valid": False,
                "error": "story must reference real object IDs from scene_inventory (e.g., 'chest_of_drawers_54', 'table_59'). Do not use generic descriptions like 'a drawer' or 'the table'.",
                "summary": "Task validation failed - story not grounded"
            }

        # Check that object IDs in story actually exist in scene
        if self.scene_data:
            valid_scene_ids = set(
                self.scene_data.rooms +
                self.scene_data.furniture +
                self.scene_data.objects
            )
            # Also allow item_ prefixed IDs (custom items defined in task)
            defined_items = {item.get("item_id") for item in task_data.get("items", []) if item.get("item_id")}
            valid_scene_ids.update(defined_items)

            invalid_story_refs = [ref for ref in object_refs if ref not in valid_scene_ids and not ref.startswith("item_")]
            if invalid_story_refs:
                return {
                    "valid": False,
                    "error": f"story references objects that don't exist in scene: {invalid_story_refs}. Use only: {list(self.scene_data.objects)[:10]}...",
                    "summary": "Task validation failed - invented object IDs"
                }

            # Also check agent_secrets for invented object IDs
            for agent_id, secrets in task_data.get("agent_secrets", {}).items():
                for secret in secrets:
                    secret_refs = re.findall(object_pattern, secret)
                    invalid_secret_refs = [ref for ref in secret_refs if ref not in valid_scene_ids and not ref.startswith("item_")]
                    if invalid_secret_refs:
                        return {
                            "valid": False,
                            "error": f"agent_secrets[{agent_id}] references objects that don't exist in scene: {invalid_secret_refs}",
                            "summary": "Task validation failed - invented object IDs in secrets"
                        }

            # Check subtask success_conditions for invented object IDs
            for subtask in task_data.get("subtasks", []):
                sc = subtask.get("success_condition", {})
                for field in ["entity", "target"]:
                    val = sc.get(field, "")
                    if val and not val.startswith("agent_"):
                        val_refs = re.findall(object_pattern, val)
                        invalid_sc_refs = [ref for ref in val_refs if ref not in valid_scene_ids and not ref.startswith("item_")]
                        if invalid_sc_refs:
                            return {
                                "valid": False,
                                "error": f"subtask '{subtask.get('id')}' success_condition references objects that don't exist: {invalid_sc_refs}",
                                "summary": "Task validation failed - invented object IDs in subtasks"
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
        Run the actual benchmark test in a subprocess.

        Uses subprocess to get a fresh GL context (avoids context corruption
        when running multiple tests in the same process).
        """
        # Write task to temp file for subprocess
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(task_data, f)
                temp_task_file = f.name
            # Also create a temp file for results
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_result_file = f.name
        except Exception as e:
            return {
                "steps": 0,
                "done": False,
                "error": f"Failed to write temp task file: {e}",
                "summary": f"Setup error: {e}"
            }

        # Run test in subprocess
        script_path = Path(__file__).parent / "test_task.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--task-file", temp_task_file,
            "--result-file", temp_result_file,
            "--config-name", "examples/emtom_2_robots",
            "--max-turns", "20",
        ]

        try:
            # Stream stderr to terminal for live logging, capture stdout
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=None,  # Inherit stderr - shows logs in terminal
                text=True,
                timeout=600,  # 10 minute timeout for LLM agents
            )

            # Read result from file (cleaner than parsing stdout)
            try:
                with open(temp_result_file) as f:
                    result = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                result = {
                    "steps": 0,
                    "done": False,
                    "error": f"Failed to read result file: {e}",
                    "summary": "Result file error"
                }

            # Clean up temp files
            for temp_file in [temp_task_file, temp_result_file]:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

            return result

        except subprocess.TimeoutExpired:
            for temp_file in [temp_task_file, temp_result_file]:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            return {
                "steps": 0,
                "done": False,
                "error": "Test timed out after 10 minutes",
                "summary": "Timeout"
            }
        except Exception as e:
            for temp_file in [temp_task_file, temp_result_file]:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
            return {
                "steps": 0,
                "done": False,
                "error": f"Subprocess error: {e}",
                "summary": f"Subprocess error: {e}"
            }

    def _verify_golden_trajectory(self) -> str:
        """
        Execute golden_trajectory step-by-step to prove task is completable.

        Runs verification in a subprocess to get a fresh GL context
        (avoids OpenGL context issues when reusing the main process).
        """
        import subprocess
        import tempfile

        # Check task file exists
        if not self.task_file.exists():
            return json.dumps({
                "valid": False,
                "error": "working_task.json does not exist. Create it first with bash."
            })

        # Load task to check golden_trajectory exists
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return json.dumps({
                "valid": False,
                "error": f"Invalid JSON in working_task.json: {e}"
            })

        golden = task_data.get("golden_trajectory", [])
        if not golden:
            return json.dumps({
                "valid": False,
                "error": "No golden_trajectory found in task. Add a golden_trajectory field with the expected action sequence."
            })

        self._log(f"Verifying golden trajectory: {len(golden)} steps")

        # Create temp file for results
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_result_file = f.name
        except Exception as e:
            return json.dumps({
                "valid": False,
                "error": f"Failed to create temp result file: {e}"
            })

        # Run verification in subprocess (fresh GL context)
        script_path = Path(__file__).parent / "verify_trajectory.py"
        cmd = [
            sys.executable,
            str(script_path),
            "--task-file", str(self.task_file),
            "--result-file", temp_result_file,
            "--config-name", "examples/emtom_2_robots",
        ]

        try:
            timeout = 1800 # 30 minutes
            # Stream stderr to terminal for live logging
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=None,  # Inherit stderr - shows logs in terminal
                text=True,
                timeout=timeout,
            )

            # Read result from file (cleaner than parsing stdout)
            try:
                with open(temp_result_file) as f:
                    result_data = json.load(f)
                if result_data.get("valid", False):
                    self.last_verify_passed = True
            except (FileNotFoundError, json.JSONDecodeError) as e:
                result_data = {
                    "valid": False,
                    "error": f"Failed to read result file: {e}",
                    "summary": "Result file error"
                }

            # Clean up temp file
            try:
                os.unlink(temp_result_file)
            except Exception:
                pass

            return json.dumps(result_data, indent=2)

        except subprocess.TimeoutExpired:
            try:
                os.unlink(temp_result_file)
            except Exception:
                pass
            return json.dumps({
                "valid": False,
                "error": f"Verification timed out after {timeout} seconds",
                "summary": "Timeout"
            })
        except Exception as e:
            try:
                os.unlink(temp_result_file)
            except Exception:
                pass
            return json.dumps({
                "valid": False,
                "error": f"Verification subprocess error: {e}",
                "summary": f"Subprocess error: {e}"
            })

    def _submit_task(self) -> str:
        """
        Copy working task to output directory.

        Called when the agent determines task quality is good.
        Requires verify_golden_trajectory[] to pass first.
        """
        if not self.last_verify_passed:
            return json.dumps({
                "error": "Must run verify_golden_trajectory[] first and pass before submitting.",
                "hint": "Run verify_golden_trajectory[] to prove the golden trajectory works."
            })

        if not self.task_file.exists():
            return "Error: working_task.json does not exist. Create and test it first."

        # Load task to validate
        try:
            with open(self.task_file) as f:
                task_data = json.load(f)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON: {e}"

        # Generate filename: {datetime}_{title_slug}.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        title = task_data.get("title", "untitled")
        # Slugify title: lowercase, replace spaces with underscores, keep only alphanumeric and underscores
        title_slug = re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:50]
        output_filename = f"{timestamp}_{title_slug}.json"
        output_path = self.output_dir / output_filename

        # Copy file
        shutil.copy(self.task_file, output_path)
        self.submitted_tasks.append(str(output_path))

        return f"Task saved to {output_path}. Total submitted: {len(self.submitted_tasks)}"

    def _fail(self, reason: str) -> str:
        """
        Mark task generation as failed.

        This should only be used for truly unrecoverable errors.
        """
        self.failed = True
        self.fail_reason = reason
        self._log(f"FAIL: {reason}")
        return f"Task generation aborted: {reason}"

    def _format_scene_data(self) -> str:
        """Format scene data for the LLM prompt."""
        if not self.scene_data:
            return "No scene data available."

        lines = []

        # Warning about exact IDs
        lines.append("**WARNING: You MUST use EXACT object IDs from this list. Do NOT invent or guess IDs!**\n")

        # Rooms - show all
        lines.append("### Rooms")
        for room in self.scene_data.rooms:
            lines.append(f"  - {room}")

        # Articulated furniture (can open/close - good for mechanics) - show all
        lines.append("\n### Articulated Furniture (can open/close)")
        for furn in self.scene_data.articulated_furniture:
            lines.append(f"  - {furn}")

        # Other furniture - show all
        other_furniture = [f for f in self.scene_data.furniture
                          if f not in self.scene_data.articulated_furniture]
        lines.append("\n### Other Furniture (tables, counters, etc.)")
        for furn in other_furniture:
            lines.append(f"  - {furn}")

        # Objects with their locations - show all
        lines.append("\n### Objects (can be picked up and placed)")
        # Build reverse mapping: object -> furniture it's on
        obj_locations = {}
        for furn, objs in self.scene_data.objects_on_furniture.items():
            for obj in objs:
                obj_locations[obj] = furn

        for obj in self.scene_data.objects:
            location = obj_locations.get(obj)
            if location:
                lines.append(f"  - {obj} (on {location})")
            else:
                lines.append(f"  - {obj}")

        return "\n".join(lines)

    def _log(self, message: str, truncate_terminal: int = 0) -> None:
        """Print log message and write to log file.

        Args:
            message: Message to log
            truncate_terminal: If > 0, truncate terminal output to this many chars
                               (log file always gets full content)
        """
        # Always write full message to log file
        if hasattr(self, '_log_handle') and self._log_handle:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._log_handle.write(f"[{timestamp}] {message}\n")
            self._log_handle.flush()

        # Print to terminal if verbose (optionally truncated)
        if self.verbose:
            if truncate_terminal > 0 and len(message) > truncate_terminal:
                print(f"{message[:truncate_terminal]}...", flush=True)
            else:
                print(message, flush=True)

    def close(self) -> None:
        """Close log file handle."""
        if hasattr(self, '_log_handle') and self._log_handle:
            self._log_handle.close()
            self._log_handle = None
