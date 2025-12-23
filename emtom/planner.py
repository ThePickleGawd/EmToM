"""
EmtomPlanner - LLMPlanner with ReAct-style logging for EMTOM benchmark.
"""

from typing import Any, Dict, Optional

from habitat_llm.planner import LLMPlanner


class EmtomPlanner(LLMPlanner):
    """LLMPlanner subclass with ReAct-style action logging."""

    AGENT_COLORS = [
        "\033[94m",  # Blue
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
    ]
    RESET = "\033[0m"

    def _log_high_level_actions(
        self,
        high_level_actions: Dict[int, Any],
        thought: Optional[str] = None,
    ) -> None:
        """
        Log high-level actions in ReAct style (Thought, Action, Observation).

        :param high_level_actions: Dict mapping agent uid to action tuple.
        :param thought: Optional thought string from LLM.
        """
        for uid, action_tuple in high_level_actions.items():
            if action_tuple and action_tuple[0]:
                action_name, action_arg, _ = action_tuple
                color = self.AGENT_COLORS[uid % len(self.AGENT_COLORS)]

                if thought:
                    print(f"Thought: {thought}", flush=True)
                print(
                    f"{color}Agent_{uid}:{self.RESET} {action_name}[{action_arg}]",
                    flush=True,
                )
