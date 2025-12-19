"""
LLM-guided curiosity model for EMTOM exploration.

Selects actions based on novelty and desire to understand the world,
rather than task-directed behavior.

Uses YAML config for prompts (like the benchmark) for consistency and scalability.
"""

from __future__ import annotations

import random
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

    IMPORTANT: Each agent gets its own LLM client to ensure independent API calls.
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
            llm_client: LLM client (kept for backward compatibility, but not used)
            instruct_config: Optional instruct config (OmegaConf). If None, loads default.
            llm_config: Optional LLM config with system_tag, user_tag, etc.
        """
        # Per-agent LLM clients (each agent gets independent API calls with different temperatures)
        self._agent_llms: Dict[str, Any] = {}

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

        # Per-agent prompts (accumulates conversation like benchmark planner)
        self.agent_prompts: Dict[str, str] = {}

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

    def _get_agent_llm(self, agent_id: str) -> Any:
        """
        Get or create an LLM client for the given agent.

        Each agent gets its own independent LLM client to ensure separate API calls.
        This prevents agents from sharing conversation history or cached responses.

        Temperature is randomized per agent AND per run to ensure different exploration
        paths each time the simulation runs (for models that support it).

        Args:
            agent_id: The agent identifier (e.g., "agent_0")

        Returns:
            LLM client for this agent
        """
        if agent_id not in self._agent_llms:
            # Create a new LLM client for this agent
            from habitat_llm.llm import instantiate_llm

            # Use different temperature for each agent to encourage diverse exploration
            agent_uid = int(agent_id.split("_")[-1]) if "_" in agent_id else 0

            # Base temperature per agent (higher values = more random)
            # Agent 0: 0.7, Agent 1: 0.8, Agent 2: 0.9, etc.
            base_temperature = 0.7 + (agent_uid * 0.1)

            # Add random offset per run to ensure different paths each simulation
            # Random offset between -0.1 and +0.1
            random_offset = random.uniform(-0.1, 0.1)
            temperature = base_temperature + random_offset

            # Clamp temperature between 0.5 and 1.0 for good randomness
            temperature = max(0.5, min(temperature, 1.0))

            self._agent_llms[agent_id] = instantiate_llm(
                "openai_chat",
                generation_params={"temperature": temperature},
            )

            # Check if model supports temperature (o1, o3, gpt-5 don't)
            model_name = self._agent_llms[agent_id].generation_params.model.lower()
            fixed_temp_models = ["o1", "o3", "gpt-5"]
            uses_fixed_temp = any(m in model_name for m in fixed_temp_models)

            # Color code: agent in color, temperature in red
            agent_color = self._get_agent_color(agent_uid)
            if uses_fixed_temp:
                print(f"[CuriosityModel] Created independent LLM client for {agent_color}{agent_id}\033[0m (model={model_name} uses fixed temperature=1)")
            else:
                print(f"[CuriosityModel] Created independent LLM client for {agent_color}{agent_id}\033[0m (temperature=\033[91m{temperature:.3f}\033[0m)")

        return self._agent_llms[agent_id]

    @staticmethod
    def _get_agent_color(agent_uid: int) -> str:
        """Get ANSI color code for an agent.

        Args:
            agent_uid: Agent UID (0, 1, 2, ...)

        Returns:
            ANSI escape code for the agent's color
        """
        colors = [
            "\033[94m",  # Blue - Agent 0
            "\033[92m",  # Green - Agent 1
            "\033[93m",  # Yellow - Agent 2
            "\033[95m",  # Magenta - Agent 3
            "\033[96m",  # Cyan - Agent 4
            "\033[91m",  # Red - Agent 5
        ]
        return colors[agent_uid % len(colors)]

    def set_tool_descriptions(self, tool_descriptions: str):
        """Set the tool descriptions to use in prompts."""
        self._tool_descriptions = tool_descriptions

    def reset(self, agent_id: Optional[str] = None):
        """Reset the conversation state for a new episode.

        Args:
            agent_id: If provided, reset only that agent. Otherwise reset all.
        """
        if agent_id:
            self.agent_prompts[agent_id] = ""
        else:
            self.agent_prompts = {}

    def select_action(
        self,
        agent_id: str,
        world_description: str,
        exploration_history: Optional[List[Dict[str, Any]]] = None,
        tool_descriptions: Optional[str] = None,
    ) -> ActionChoice:
        """
        Select an action based on curiosity.

        Args:
            agent_id: ID of the agent selecting
            world_description: Text description of current world state
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
        tools_desc = tool_descriptions or self._tool_descriptions
        if not tools_desc:
            raise ValueError(
                "Tool descriptions not set. Call set_tool_descriptions() or pass tool_descriptions parameter."
            )

        # Build task description from world state (with agent-specific exploration focus)
        task = self._build_task_description(agent_id, world_description, exploration_history or [])

        # Build initial prompt if this is the first action for this agent
        if agent_id not in self.agent_prompts or not self.agent_prompts[agent_id]:
            self.agent_prompts[agent_id] = self.prompt_template.format(
                id=agent_uid,
                tool_descriptions=tools_desc,
                input=task,
            )

        # Get this agent's dedicated LLM client (ensures independent API calls)
        agent_llm = self._get_agent_llm(agent_id)

        # Generate response using this agent's prompt and LLM
        curr_prompt = self.agent_prompts[agent_id]
        response = agent_llm.generate(curr_prompt, self.stopword)

        # Parse response - fail explicitly if it doesn't match expected format
        action_choice = self._parse_response(response, agent_uid)

        # Append response to this agent's conversation
        curr_prompt += response
        if not curr_prompt.endswith(self.stopword):
            curr_prompt += self.stopword
        self.agent_prompts[agent_id] = curr_prompt

        return action_choice

    def add_observation(self, agent_id: str, observation: str):
        """
        Add an observation to this agent's conversation.

        Args:
            agent_id: ID of the agent
            observation: The observation text
        """
        agent_uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id
        if agent_id not in self.agent_prompts:
            self.agent_prompts[agent_id] = ""
        self.agent_prompts[agent_id] += f"\nAgent_{agent_uid}_Observation: {observation}\n"

    def get_all_prompts(self) -> Dict[str, str]:
        """Get all agent prompts for saving.

        Returns:
            Dict mapping agent_id to their full prompt/conversation history
        """
        return self.agent_prompts.copy()

    def _build_task_description(
        self,
        agent_id: str,
        world_description: str,
        exploration_history: List[Dict[str, Any]],
    ) -> str:
        """Build a task description from current world state.

        IMPORTANT: Each agent gets a unique exploration directive to ensure
        they explore different parts of the environment independently.
        This prevents all agents from converging on the same objects.

        Args:
            agent_id: The agent identifier (e.g., "agent_0")
            world_description: Text description of the world state
            exploration_history: Recent actions for context

        Returns:
            Task description string for the LLM prompt
        """
        # Extract agent UID for assigning unique exploration focus
        agent_uid = int(agent_id.split("_")[-1]) if "_" in agent_id else 0

        # Generate unique exploration directive for each agent
        # This ensures agents explore different areas/object types
        exploration_directive = self._get_agent_exploration_directive(agent_uid, world_description)

        parts = [exploration_directive]
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

    def _get_agent_exploration_directive(self, agent_uid: int, world_description: str) -> str:
        """
        Generate a unique exploration directive for each agent.

        This is CRITICAL for ensuring agents explore independently rather than
        converging on the same objects. Each agent gets:
        1. Their actual spawn room location
        2. Instructions to use random sampling for action selection
        3. A unique exploration focus area

        Args:
            agent_uid: The agent's numeric ID (0, 1, 2, ...)
            world_description: The world state description (to extract room/location info)

        Returns:
            A unique exploration directive string
        """
        # Extract current location from world description
        current_room = "unknown"
        if "You are in " in world_description:
            # Extract room name from "You are in <room>."
            start = world_description.find("You are in ") + len("You are in ")
            end = world_description.find(".", start)
            if end > start:
                current_room = world_description[start:end].strip()

        # Extract available rooms from world description
        rooms = []
        if "Rooms you can go to:" in world_description:
            rooms_line = world_description.split("Rooms you can go to:")[1].split("\n")[0]
            rooms = [r.strip() for r in rooms_line.split(",")]

        # Extract furniture from world description for action targets
        furniture = []
        if "Furniture:" in world_description:
            furniture_line = world_description.split("Furniture:")[1].split("\n")[0]
            furniture = [f.strip() for f in furniture_line.split(",")]

        # Define the available actions for random sampling
        available_actions = [
            "Navigate[<room or furniture>]",
            "Open[<furniture>]",
            "Close[<furniture>]",
            "Explore[<room>]",
            "Pick[<object>]",
            "Place[<object>, <receptacle>]",
            "FindObjectTool[<query>]",
            "FindReceptacleTool[<query>]",
            "FindRoomTool[<query>]",
        ]

        # Build the directive with spawn location and RNG instructions
        directive = f"""You are Agent {agent_uid}. You have SPAWNED in: {current_room}.

CRITICAL - RANDOM ACTION SELECTION:
You MUST use random sampling (RNG) to select your actions. Do NOT always pick the "most logical" or "first" option.
To decide what to do:
1. Mentally list all valid actions: {', '.join(available_actions)}
2. Randomly pick ONE action type (imagine rolling a dice)
3. Randomly pick a target from the available options (imagine rolling another dice)
4. Execute that randomly selected action

Available action types to randomly sample from:
- Navigate: Go to a room or furniture piece
- Open/Close: Interact with openable furniture (cabinets, drawers, fridges, etc.)
- Explore: Search a room thoroughly
- Pick/Place: Pick up or put down objects
- FindObjectTool/FindReceptacleTool/FindRoomTool: Search for objects

Your current location: {current_room}
Rooms you can randomly navigate to: {', '.join(rooms) if rooms else 'use FindRoomTool to discover'}
Furniture you can randomly interact with: {', '.join(furniture[:8]) if furniture else 'use FindReceptacleTool to discover'}

EXPLORATION GOAL: Discover interesting and unexpected object behaviors (surprises).
When something unexpected happens, note it with "Surprise:" in your Thought.

IMPORTANT RULES:
- Use RNG mentally to pick actions - do NOT always choose the same things
- Each turn, genuinely randomize your choice from valid options
- Do NOT repeat the same action on the same target consecutively
- Explore DIFFERENT areas (rooms) than your recent history shows
- In every room that you are in, look around at object you think could result in generating a "suprise" """

        return directive

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
