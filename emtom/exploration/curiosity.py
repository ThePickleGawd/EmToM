"""
LLM-guided curiosity model for EMTOM exploration.

Selects actions based on novelty and desire to understand the world,
rather than task-directed behavior.

Uses YAML config for prompts (like the benchmark) for consistency and scalability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


@dataclass
class ActionChoice:
    """Result of curiosity-driven action selection."""

    action: str
    target: Optional[str]
    reasoning: str
    surprise: Optional[str] = None  # If LLM detected something unexpected


class CuriosityModel:
    """
    LLM-guided action selection based on novelty and exploration.

    Uses an LLM to select actions that are likely to reveal new information
    about how the world works, particularly to discover unexpected behaviors.

    Uses YAML config for prompts (matching benchmark structure) for consistency.
    """

    # Default config path relative to habitat_llm/conf/instruct/
    DEFAULT_CONFIG = "emtom_exploration"

    def __init__(
        self,
        llm_client: Any,
        instruct_config: Optional[Any] = None,
        llm_config: Optional[Any] = None,
    ):
        """
        Initialize the curiosity model.

        Args:
            llm_client: LLM client with generate(prompt) method
            instruct_config: Optional instruct config (OmegaConf). If None, loads default.
            llm_config: Optional LLM config with system_tag, user_tag, etc.
        """
        self.llm = llm_client

        # Load instruct config if not provided
        if instruct_config is not None:
            self.instruct = instruct_config
        else:
            self.instruct = self._load_default_config()

        # Extract prompt template and other settings
        self.prompt_template = self.instruct.prompt
        self.stopword = self.instruct.get("stopword", "Assigned!")
        self.end_expression = self.instruct.get("end_expression", "Final Thought:")

        # LLM config for tags (defaults if not provided)
        self.llm_config = llm_config or OmegaConf.create({
            "system_tag": "",
            "user_tag": "",
            "assistant_tag": "",
            "eot_tag": "",
        })

        # Current prompt (accumulates conversation like benchmark planner)
        self.curr_prompt = ""

        # Tool descriptions (set by explorer)
        self._tool_descriptions: Optional[str] = None

    def _load_default_config(self) -> Any:
        """Load the default exploration YAML config."""
        config_path = Path(__file__).parent.parent.parent / "habitat_llm" / "conf" / "instruct" / f"{self.DEFAULT_CONFIG}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Exploration config not found at {config_path}. "
                "Please ensure emtom_exploration.yaml exists."
            )

        return OmegaConf.load(config_path)

    def set_tool_descriptions(self, tool_descriptions: str):
        """Set the tool descriptions to use in prompts."""
        self._tool_descriptions = tool_descriptions

    def reset(self):
        """Reset the conversation state for a new episode."""
        self.curr_prompt = ""

    def select_action(
        self,
        agent_id: str,
        world_description: str,
        available_actions: List[Dict[str, Any]],
        exploration_history: Optional[List[Dict[str, Any]]] = None,
        tool_descriptions: Optional[str] = None,
    ) -> ActionChoice:
        """
        Select an action based on curiosity.

        Args:
            agent_id: ID of the agent selecting
            world_description: Text description of current world state
            available_actions: List of available actions with targets
            exploration_history: Recent action history for context
            tool_descriptions: Optional tool descriptions (uses stored if not provided)

        Returns:
            ActionChoice with selected action, reasoning, and optional surprise

        Raises:
            ValueError: If the LLM response doesn't match expected ReACT format
        """
        # Extract agent UID (e.g., "agent_0" -> "0")
        agent_uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id

        # Use provided tool descriptions or stored ones
        tools_desc = tool_descriptions or self._tool_descriptions or self._get_default_tool_descriptions()

        # Build task description from world state
        task = self._build_task_description(world_description, exploration_history or [])

        # Build initial prompt if this is the first action
        if not self.curr_prompt:
            self.curr_prompt = self.prompt_template.format(
                id=agent_uid,
                tool_descriptions=tools_desc,
                input=task,
            )

        # Generate response
        response = self.llm.generate(self.curr_prompt, self.stopword)

        # Parse response - fail explicitly if it doesn't match expected format
        action_choice = self._parse_response(response, agent_uid)

        # Append response to conversation
        self.curr_prompt += response
        if not self.curr_prompt.endswith(self.stopword):
            self.curr_prompt += self.stopword

        return action_choice

    def add_observation(self, agent_id: str, observation: str):
        """
        Add an observation to the conversation.

        Args:
            agent_id: ID of the agent
            observation: The observation text
        """
        agent_uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id
        self.curr_prompt += f"\nAgent_{agent_uid}_Observation: {observation}\n"

    def _build_task_description(
        self,
        world_description: str,
        exploration_history: List[Dict[str, Any]],
    ) -> str:
        """Build a task description from current world state."""
        parts = ["Explore the house and discover interesting object behaviors."]
        parts.append(f"\nCurrent state: {world_description}")

        if exploration_history:
            parts.append("\nRecent actions:")
            for entry in exploration_history[-3:]:
                action = entry.get("action", "unknown")
                target = entry.get("target", "")
                obs = entry.get("observation", "")
                obs_short = obs[:50] + "..." if len(obs) > 50 else obs
                if target:
                    parts.append(f"  - {action}[{target}]: {obs_short}")
                else:
                    parts.append(f"  - {action}: {obs_short}")

        return "\n".join(parts)

    def _get_default_tool_descriptions(self) -> str:
        """Get default tool descriptions."""
        return """Available tools:
- Navigate[target]: Move to a room or furniture. You MUST navigate close to objects before interacting.
- Open[furniture]: Open articulated furniture (cabinets, drawers, fridges). Must be near it first.
- Close[furniture]: Close articulated furniture. Must be near it first.
- Pick[object]: Pick up an object. Must be near it first.
- Explore[room]: Thoroughly search a room by visiting all furniture in it.
- FindObjectTool[query]: Search for objects matching a description.
- FindReceptacleTool[query]: Search for furniture/receptacles matching a description.
- FindRoomTool[query]: Search for rooms matching a description.
- FindAgentActionTool[]: Check what the other agent is doing.
- Inspect[object]: Carefully examine an object to learn its properties and state.
- Hide[object]: Hide an object so others can't see it.
- Communicate[message]: Send a message to your teammate."""

    def _parse_response(self, response: str, agent_uid: str) -> ActionChoice:
        """
        Parse LLM response into ActionChoice.

        Extracts:
        - Reasoning (the Thought text)
        - Surprise (if "Surprise:" is in the Thought)
        - Action and target

        Args:
            response: The LLM response text
            agent_uid: The agent UID (e.g., "0")

        Returns:
            ActionChoice with action, target, reasoning, and optional surprise

        Raises:
            ValueError: If the response doesn't match expected format
        """
        # Check for Final Thought (end of exploration)
        if "Final Thought:" in response:
            # Extract reasoning before Final Thought
            parts = response.split("Final Thought:")
            reasoning = parts[0].strip()
            # Remove "Thought:" prefix if present
            if reasoning.startswith("Thought:"):
                reasoning = reasoning[8:].strip()
            # Check for surprise in the reasoning
            surprise = self._extract_surprise(reasoning)
            return ActionChoice(
                action="Done",
                target=None,
                reasoning=reasoning if reasoning else "Exploration complete",
                surprise=surprise,
            )

        # Extract Action - try formats in order of preference
        # Note: Use greedy (.+) to handle nested brackets in targets
        # e.g., Pick[key [#127]] should capture "key [#127]" as the full target
        # The greedy match will find the LAST closing char on the line
        # Also handle LLM typos like ) instead of ] for closing bracket

        # 1. Agent_{id}_Action: ActionName[target]
        action_pattern = rf"Agent_{agent_uid}_Action:\s*(\w+)\[(.+)[\]\)]"
        action_match = re.search(action_pattern, response)

        # 2. Agent_X_Action: ActionName[target] (any agent ID)
        if not action_match:
            action_pattern = r"Agent_\d+_Action:\s*(\w+)\[(.+)[\]\)]"
            action_match = re.search(action_pattern, response)

        # 3. Simple ActionName[target] format (LLM sometimes omits prefix)
        if not action_match:
            action_pattern = r"^(\w+)\[(.+)[\]\)]"
            action_match = re.search(action_pattern, response.strip(), re.MULTILINE)

        if not action_match:
            raise ValueError(
                f"Failed to parse Action from LLM response. "
                f"Expected 'Agent_{agent_uid}_Action: ActionName[target]' but got:\n{response}"
            )

        action_name = action_match.group(1)
        target = action_match.group(2) if action_match.group(2) else None

        # Extract reasoning - everything before the Action
        action_start = action_match.start()
        reasoning = response[:action_start].strip()

        # Remove "Thought:" prefix if present
        if reasoning.startswith("Thought:"):
            reasoning = reasoning[8:].strip()

        # Extract surprise from reasoning
        surprise = self._extract_surprise(reasoning)

        # If no reasoning found, use a default
        if not reasoning:
            reasoning = f"Executing {action_name}"

        return ActionChoice(
            action=action_name,
            target=target,
            reasoning=reasoning,
            surprise=surprise,
        )

    def _extract_surprise(self, reasoning: str) -> Optional[str]:
        """
        Extract surprise description from reasoning if present.

        Looks for "Surprise:" keyword in the reasoning text.

        Args:
            reasoning: The thought/reasoning text

        Returns:
            The surprise description, or None if no surprise detected
        """
        # Look for "Surprise:" in the reasoning
        surprise_match = re.search(r"Surprise:\s*(.+?)(?=\.|!|$)", reasoning, re.IGNORECASE)
        if surprise_match:
            return surprise_match.group(1).strip()

        # Also check for variations
        surprise_match = re.search(r"Surprise:\s*(.+)", reasoning, re.IGNORECASE)
        if surprise_match:
            return surprise_match.group(1).strip()

        return None
