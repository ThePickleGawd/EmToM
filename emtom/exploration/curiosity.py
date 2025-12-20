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

from emtom.exploration.verbalized_sampling import (
    VerbalizedSampler,
    VerbalizedDistribution,
)


@dataclass
class ActionChoice:
    """Result of curiosity-driven action selection."""

    action: str
    target: Optional[str]
    reasoning: str
    surprise: Optional[str] = None  # If LLM detected something unexpected
    vs_distribution: Optional[VerbalizedDistribution] = None  # Verbalized sampling distribution


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

    # Config for verbalized sampling prompt
    VS_CONFIG = "emtom_verbalized_sampling"

    def __init__(
        self,
        llm_client: Any,
        instruct_config: Optional[Any] = None,
        llm_config: Optional[Any] = None,
        use_verbalized_sampling: bool = True,
    ):
        """
        Initialize the curiosity model.

        Args:
            llm_client: LLM client (kept for backward compatibility, but not used)
            instruct_config: Optional instruct config (OmegaConf). If None, loads default.
            llm_config: Optional LLM config with system_tag, user_tag, etc.
            use_verbalized_sampling: If True, use Verbalized Sampling (VS) for action selection.
                                    VS prompts the LLM to output probability distributions
                                    over actions, then samples from them programmatically.
        """
        # Per-agent LLM clients (each agent gets independent API calls with different temperatures)
        self._agent_llms: Dict[str, Any] = {}

        # Verbalized Sampling settings
        self.use_verbalized_sampling = use_verbalized_sampling
        self._vs_sampler = VerbalizedSampler() if use_verbalized_sampling else None
        self._agent_rngs: Dict[str, random.Random] = {}  # Per-agent RNGs for sampling

        # Load instruct config if not provided
        if instruct_config is not None:
            self.instruct = instruct_config
        else:
            # Load VS config if using verbalized sampling, otherwise default
            if use_verbalized_sampling:
                self.instruct = self._load_vs_config()
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

        # Story context from scenario system (set by explorer)
        self._story_context: Optional[str] = None

        # Available actions cache (set by explorer for VS)
        self._available_actions: Optional[List[Dict[str, Any]]] = None

        if use_verbalized_sampling:
            print("[CuriosityModel] Using Verbalized Sampling (VS) for action selection")

    def _load_default_config(self) -> Any:
        """Load the default exploration YAML config."""
        config_path = Path(__file__).parent.parent.parent / "habitat_llm" / "conf" / "instruct" / f"{self.DEFAULT_CONFIG}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(
                f"Exploration config not found at {config_path}. "
                "Please ensure emtom_exploration.yaml exists."
            )

        return OmegaConf.load(config_path)

    def _load_vs_config(self) -> Any:
        """Load the Verbalized Sampling YAML config."""
        config_path = Path(__file__).parent.parent.parent / "habitat_llm" / "conf" / "instruct" / f"{self.VS_CONFIG}.yaml"

        if not config_path.exists():
            print(f"[CuriosityModel] VS config not found at {config_path}, falling back to default")
            return self._load_default_config()

        return OmegaConf.load(config_path)

    def _get_agent_rng(self, agent_id: str) -> random.Random:
        """
        Get or create an RNG for the given agent.

        Each agent gets a unique RNG seeded by their ID to ensure
        reproducible but different sampling behavior.

        Args:
            agent_id: The agent identifier (e.g., "agent_0")

        Returns:
            random.Random instance for this agent
        """
        if agent_id not in self._agent_rngs:
            # Create agent-specific seed from agent_id
            agent_uid = int(agent_id.split("_")[-1]) if "_" in agent_id else 0
            # Use current time + agent_uid for unique but reproducible seed per run
            import time
            seed = int(time.time() * 1000) + agent_uid * 12345
            self._agent_rngs[agent_id] = random.Random(seed)
            print(f"[CuriosityModel] Created RNG for {agent_id} with seed {seed}")
        return self._agent_rngs[agent_id]

    def set_available_actions(self, actions: List[Dict[str, Any]]):
        """Set the available actions for Verbalized Sampling.

        Args:
            actions: List of action dictionaries with 'name' and 'targets' keys
        """
        self._available_actions = actions

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

    def set_story_context(self, story_context: str):
        """Set the story context from scenario system to use in prompts."""
        self._story_context = story_context

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
        available_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> ActionChoice:
        """
        Select an action based on curiosity.

        When use_verbalized_sampling is True, uses Verbalized Sampling (VS):
        1. Prompts the LLM to output a probability distribution over actions
        2. Parses the distribution
        3. Samples from it using the agent's RNG

        Args:
            agent_id: ID of the agent selecting
            world_description: Text description of current world state
            exploration_history: Recent action history for context
            tool_descriptions: Optional tool descriptions (uses stored if not provided)
            available_actions: Optional list of available actions for VS (uses stored if not provided)

        Returns:
            ActionChoice with selected action, reasoning, and optional surprise

        Raises:
            ValueError: If the LLM response doesn't match expected ReACT format
        """
        # Use Verbalized Sampling if enabled
        if self.use_verbalized_sampling:
            return self._select_action_vs(
                agent_id=agent_id,
                world_description=world_description,
                exploration_history=exploration_history,
                tool_descriptions=tool_descriptions,
                available_actions=available_actions,
            )

        # --- Original logic (non-VS) ---
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

    def _select_action_vs(
        self,
        agent_id: str,
        world_description: str,
        exploration_history: Optional[List[Dict[str, Any]]] = None,
        tool_descriptions: Optional[str] = None,
        available_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> ActionChoice:
        """
        Select an action using Verbalized Sampling (VS).

        This method:
        1. Prompts the LLM to output a probability distribution over actions
        2. Parses the distribution from the response
        3. Samples from it using the agent's dedicated RNG

        Args:
            agent_id: ID of the agent selecting
            world_description: Text description of current world state
            exploration_history: Recent action history for context
            tool_descriptions: Optional tool descriptions
            available_actions: List of available actions with targets

        Returns:
            ActionChoice with selected action, reasoning, distribution, and optional surprise
        """
        agent_uid = agent_id.split("_")[-1] if "_" in agent_id else agent_id
        agent_uid_int = int(agent_uid) if agent_uid.isdigit() else 0
        agent_color = self._get_agent_color(agent_uid_int)
        reset = "\033[0m"

        # Use provided tool descriptions or stored ones
        tools_desc = tool_descriptions or self._tool_descriptions
        if not tools_desc:
            raise ValueError(
                "Tool descriptions not set. Call set_tool_descriptions() or pass tool_descriptions parameter."
            )

        # Use provided actions or stored ones
        actions = available_actions or self._available_actions
        if not actions:
            # Build default action list from world description
            actions = self._build_default_actions(world_description)

        # Get this agent's dedicated LLM client and RNG
        agent_llm = self._get_agent_llm(agent_id)
        agent_rng = self._get_agent_rng(agent_id)

        # Build the VS prompt
        vs_prompt = self._vs_sampler.build_distribution_prompt(
            available_actions=actions,
            world_description=world_description,
            exploration_history=exploration_history,
            agent_id=agent_id,
        )

        # Add story context if available
        if self._story_context:
            vs_prompt = f"== SCENARIO ==\n{self._story_context}\n\n{vs_prompt}"

        # Store prompt for logging
        if agent_id not in self.agent_prompts or not self.agent_prompts[agent_id]:
            self.agent_prompts[agent_id] = vs_prompt

        # Generate distribution from LLM
        response = agent_llm.generate(vs_prompt, stop=None)

        # Parse the distribution
        distribution = self._vs_sampler.parse_distribution(response)

        # Handle empty distribution (fallback with varied probabilities)
        if not distribution.actions:
            print(f"{agent_color}[VS]{reset} Warning: Could not parse distribution, using varied fallback")
            from emtom.exploration.verbalized_sampling import ActionProbability
            fallback_actions = []

            # Generate varied probabilities for 25 actions
            # Higher probabilities for first actions (more promising), lower for later ones
            varied_probs = [
                0.11, 0.10, 0.09, 0.08, 0.07,  # High priority (5)
                0.06, 0.06, 0.05, 0.05, 0.05,  # Medium priority (5)
                0.04, 0.04, 0.03, 0.03, 0.03,  # Lower priority (5)
                0.02, 0.02, 0.02, 0.02, 0.02,  # Low priority (5)
                0.01, 0.01, 0.01, 0.01, 0.01,  # Tail (5)
            ]

            prob_idx = 0
            for action in actions:
                name = action.get("name", "Navigate")
                targets = action.get("targets", [])
                if targets:
                    for target in targets[:3]:  # Add up to 3 targets per action type
                        if prob_idx < 25:
                            fallback_actions.append(ActionProbability(
                                action=name,
                                target=target,
                                probability=varied_probs[prob_idx],
                            ))
                            prob_idx += 1
                if prob_idx >= 25:
                    break
            distribution.actions = fallback_actions

        # Normalize probabilities to sum to 1.0 (preserves relative proportions)
        # This ensures sampling is PROPORTIONAL to LLM's output probabilities
        # e.g., if LLM outputs 0.15 for action A and 0.04 for action C,
        # A gets picked ~15% of the time, C gets picked ~4% of the time
        distribution.normalize()

        # Sample from the distribution using agent's RNG
        # Sampling is PROPORTIONAL to probabilities - if action has 0.15 prob, it's picked 15% of time
        # Even low-probability "tail" actions can be sampled
        sampled = distribution.sample(rng=agent_rng)

        # Sort actions by probability for display
        sorted_actions = sorted(distribution.actions, key=lambda a: a.probability, reverse=True)

        # Color codes for probability bars
        green = "\033[92m"
        yellow = "\033[93m"
        cyan = "\033[96m"
        magenta = "\033[95m"
        bold = "\033[1m"

        # Print colorful VS distribution header
        print(f"\n{agent_color}{bold}{'─'*70}{reset}")
        print(f"{agent_color}{bold}[VS] {agent_id.upper()} - {len(distribution.actions)} actions{reset}")
        print(f"{agent_color}{bold}{'─'*70}{reset}")

        # Print each action with colored probability bar
        for a in sorted_actions:
            bar_len = int(a.probability * 50)  # Scale bar to 50 chars for better visibility

            # Color the bar based on probability level
            if a.probability >= 0.10:
                bar_color = green
            elif a.probability >= 0.05:
                bar_color = yellow
            else:
                bar_color = cyan

            bar = "█" * bar_len

            # Highlight the sampled action with bold and different formatting
            if a.action == sampled.action and a.target == sampled.target:
                print(f"  {agent_color}{bold}{a.probability:.3f}{reset} [{bar_color}{bar:50s}{reset}] {agent_color}{bold}{a.full_action}{reset} {magenta}◄── SAMPLED{reset}")
            else:
                print(f"  {a.probability:.3f} [{bar_color}{bar:50s}{reset}] {a.full_action}")

        print(f"\n{agent_color}{bold}[VS] {agent_id} → {sampled.full_action}{reset}")
        print(f"{agent_color}{'─'*70}{reset}\n")

        # Build reasoning that includes distribution info
        top_actions = sorted_actions[:5]
        dist_summary = " | ".join(f"{a.full_action}:{a.probability:.2f}" for a in top_actions)

        # Build full reasoning
        reasoning = distribution.reasoning if distribution.reasoning else "Exploring the environment."
        full_reasoning = f"{reasoning} [VS: {dist_summary}] -> Sampled: {sampled.full_action}"

        # Check for surprise in the reasoning
        surprise = self._extract_surprise(reasoning)

        # Append to conversation for logging
        action_line = f"\nAgent_{agent_uid}_Action: {sampled.full_action}\nAssigned!\n"
        self.agent_prompts[agent_id] += f"\n{response}{action_line}"

        return ActionChoice(
            action=sampled.action,
            target=sampled.target,
            reasoning=full_reasoning,
            surprise=surprise,
            vs_distribution=distribution,
        )

    def _build_default_actions(self, world_description: str) -> List[Dict[str, Any]]:
        """Build a default action list from world description for VS fallback.

        Args:
            world_description: Text description of the world state

        Returns:
            List of action dictionaries with 'name' and 'targets' keys
        """
        actions = []

        # Extract rooms
        rooms = []
        if "Rooms you can go to:" in world_description:
            rooms_line = world_description.split("Rooms you can go to:")[1].split("\n")[0]
            rooms = [r.strip() for r in rooms_line.split(",")]

        # Extract furniture
        furniture = []
        if "Furniture:" in world_description:
            furniture_line = world_description.split("Furniture:")[1].split("\n")[0]
            furniture = [f.strip() for f in furniture_line.split(",")]

        # Extract objects
        objects = []
        if "Objects:" in world_description:
            objects_line = world_description.split("Objects:")[1].split("\n")[0]
            objects = [o.strip().split(" (")[0] for o in objects_line.split(",")]  # Remove location info

        # Build action list
        if rooms:
            actions.append({"name": "Navigate", "targets": rooms[:5]})
            actions.append({"name": "Explore", "targets": rooms[:3]})

        if furniture:
            actions.append({"name": "Navigate", "targets": furniture[:5]})
            actions.append({"name": "Open", "targets": furniture[:5]})
            actions.append({"name": "Close", "targets": furniture[:3]})

        if objects:
            actions.append({"name": "Pick", "targets": objects[:5]})

        # Always include search tools
        actions.append({"name": "FindRoomTool", "targets": ["all rooms"]})
        actions.append({"name": "FindReceptacleTool", "targets": ["furniture", "openable furniture"]})
        actions.append({"name": "FindObjectTool", "targets": ["small objects", "interesting objects"]})

        return actions

    def _build_task_description(
        self,
        agent_id: str,
        world_description: str,
        exploration_history: List[Dict[str, Any]],
    ) -> str:
        """Build a task description from current world state and story context.

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
        parts = []

        # Extract agent UID
        agent_uid = int(agent_id.split("_")[-1]) if "_" in agent_id else 0

        # Story context provides atmosphere/setting (not a task)
        if self._story_context:
            parts.append("== SCENARIO ==")
            parts.append(self._story_context)
            parts.append("")

        # Always include free exploration instructions
        parts.append("== YOUR MISSION ==")
        parts.append(f"You are Agent {agent_uid}. Explore this house freely - visit ALL rooms and interact with objects.")
        parts.append("There is no specific task. Discover interesting behaviors, hidden items, and unusual mechanics.")
        parts.append("Use random action selection to ensure broad coverage of the environment.")

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
