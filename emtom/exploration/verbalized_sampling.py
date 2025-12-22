"""
Verbalized Sampling (VS) for EMTOM exploration.

A training-free prompting strategy to circumvent mode collapse in LLM-guided exploration.
VS prompts the model to verbalize a probability distribution over possible actions,
then samples from that distribution programmatically.

This ensures true randomness in action selection while leveraging the LLM's world understanding.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ActionProbability:
    """An action with its associated probability."""
    action: str
    target: Optional[str]
    probability: float

    @property
    def full_action(self) -> str:
        """Return the full action string like 'Navigate[kitchen]'."""
        if self.target:
            return f"{self.action}[{self.target}]"
        return self.action


@dataclass
class VerbalizedDistribution:
    """A probability distribution over actions verbalized by the LLM."""
    actions: List[ActionProbability]
    reasoning: str
    raw_response: str

    def normalize(self) -> "VerbalizedDistribution":
        """Normalize probabilities to sum to 1.0."""
        total = sum(a.probability for a in self.actions)
        if total > 0:
            for a in self.actions:
                a.probability /= total
        return self

    def clamp_and_normalize(self, max_prob: float = 0.10) -> "VerbalizedDistribution":
        """
        Clamp all probabilities to be at most max_prob, then normalize.

        This ensures no single action dominates the distribution,
        encouraging more uniform exploration.

        Args:
            max_prob: Maximum allowed probability (default 0.10)

        Returns:
            Self with clamped and normalized probabilities
        """
        # First clamp any probabilities that exceed max_prob
        was_clamped = False
        for a in self.actions:
            if a.probability > max_prob:
                a.probability = max_prob
                was_clamped = True

        if was_clamped:
            print(f"[VS] Clamped probabilities to max {max_prob}")

        # Now normalize to sum to 1.0
        return self.normalize()

    def sample(self, rng: Optional[random.Random] = None) -> ActionProbability:
        """Sample an action from the distribution.

        Args:
            rng: Optional random.Random instance for reproducibility

        Returns:
            Sampled ActionProbability
        """
        if not self.actions:
            raise ValueError("Cannot sample from empty distribution")

        rng = rng or random.Random()

        # Build cumulative distribution
        cumulative = []
        running_sum = 0.0
        for action in self.actions:
            running_sum += action.probability
            cumulative.append(running_sum)

        # Sample
        r = rng.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                return self.actions[i]

        # Fallback to last action (handles floating point errors)
        return self.actions[-1]

    def get_top_k(self, k: int = 3) -> List[ActionProbability]:
        """Get the top-k most probable actions."""
        return sorted(self.actions, key=lambda a: a.probability, reverse=True)[:k]


class VerbalizedSampler:
    """
    Implements Verbalized Sampling for action selection.

    The LLM is prompted to output a probability distribution over actions,
    which is then sampled from programmatically to select the next action.
    """

    # Regex patterns for parsing probability distributions
    # Matches formats like:
    # - "Navigate[kitchen]: 0.3"
    # - "Open[fridge_0] - 0.25"
    # - "Pick[apple] (0.15)"
    # - "Explore[bedroom]: 30%"
    PROB_PATTERNS = [
        # Action[target]: probability
        r"(\w+)\[([^\]]+)\]\s*[:=\-]\s*(\d*\.?\d+)%?",
        # Action[target] (probability)
        r"(\w+)\[([^\]]+)\]\s*\((\d*\.?\d+)%?\)",
        # probability: Action[target]
        r"(\d*\.?\d+)%?\s*[:=\-]\s*(\w+)\[([^\]]+)\]",
    ]

    def __init__(self, temperature_boost: float = 0.0):
        """
        Initialize the verbalized sampler.

        Args:
            temperature_boost: Additional temperature to add when generating distributions.
                              Higher values encourage more uniform distributions.
        """
        self.temperature_boost = temperature_boost

    def build_distribution_prompt(
        self,
        available_actions: List[Dict[str, Any]],
        world_description: str,
        exploration_history: Optional[List[Dict[str, Any]]] = None,
        agent_id: str = "agent_0",
        tool_descriptions: Optional[str] = None,
    ) -> str:
        """
        Build a prompt that asks the LLM to output a probability distribution.

        The agent must use perception tools (FindObjectTool, FindReceptacleTool, etc.)
        to discover what objects/furniture exist before interacting with them.

        Args:
            available_actions: List of available actions (NOT used - for backwards compat)
            world_description: Current world state description
            exploration_history: Recent action history
            agent_id: The agent ID
            tool_descriptions: Tool descriptions from PARTNR (required for discovery-based exploration)

        Returns:
            Prompt string asking for probability distribution
        """
        # Format history
        history_text = ""
        if exploration_history:
            history_lines = []
            for entry in exploration_history[-5:]:
                action = entry.get("action", "unknown")
                target = entry.get("target", "")
                if target:
                    history_lines.append(f"  - {action}[{target}]")
                else:
                    history_lines.append(f"  - {action}")
            history_text = "Recent actions:\n" + "\n".join(history_lines)

        # Extract agent number for display
        agent_num = agent_id.split("_")[-1] if "_" in agent_id else agent_id

        # Use tool descriptions if provided, otherwise use default action descriptions
        if tool_descriptions:
            tools_section = f"""AVAILABLE TOOLS:
{tool_descriptions}

HOW TO EXPLORE:
1. Use perception tools to DISCOVER what exists:
   - FindObjectTool[query]: Find objects matching a description (e.g., "small objects", "containers")
   - FindReceptacleTool[query]: Find furniture/receptacles (e.g., "openable furniture", "tables")
   - FindRoomTool[query]: Find rooms (e.g., "all rooms", "bedrooms")

2. Once you discover specific objects, use motor skills:
   - Navigate[target]: Go to a room or furniture
   - Open[furniture]: Open articulated furniture (cabinets, drawers, fridges)
   - Close[furniture]: Close articulated furniture
   - Pick[object]: Pick up an object
   - Place[object, on, furniture, None, None]: Place an object on furniture

IMPORTANT: You must use FindObjectTool/FindReceptacleTool to discover objects before interacting with them!"""
        else:
            tools_section = """AVAILABLE TOOLS:
- FindObjectTool[query]: Discover objects matching a description
- FindReceptacleTool[query]: Discover furniture/receptacles
- FindRoomTool[query]: Discover rooms
- Navigate[target]: Go to a room or furniture piece
- Open[furniture]: Open articulated furniture (cabinets, drawers, etc.)
- Close[furniture]: Close articulated furniture
- Pick[object]: Pick up an object
- Explore[room]: Search a room thoroughly

IMPORTANT: Use FindObjectTool/FindReceptacleTool to discover specific object names before using Pick/Open!"""

        prompt = f"""You are Agent {agent_num} exploring an environment. Your goal is to DISCOVER interesting behaviors and hidden mechanics.

CURRENT STATE:
{world_description}

{history_text}

{tools_section}

VERBALIZED SAMPLING TASK:
Generate 5 candidate actions, each with a probability YOU assign based on how promising each action is for discovering interesting behaviors. The system will randomly sample ONE action from your distribution.

Assign probabilities based on:
- Using perception tools (FindObjectTool, FindReceptacleTool) to discover what exists
- Exploring rooms you haven't visited
- Interacting with furniture that might have hidden mechanics
- Your intuition about where surprises might be found

FORMAT (generate 5 responses):
<response>
<action>ActionName[target]</action>
<probability>your_assigned_probability</probability>
</response>

Example actions:
- FindObjectTool[containers] (discover container objects)
- FindReceptacleTool[openable furniture] (find things to open)
- Navigate[kitchen_1] (go to a room)
- Open[cabinet_36] (open specific furniture you discovered)

Generate your 5 candidate actions with probabilities now:"""

        return prompt

    def parse_distribution(self, response: str) -> VerbalizedDistribution:
        """
        Parse a probability distribution from the LLM response.

        Supports multiple formats:
        1. XML format: <response><action>ActionName[target]</action><probability>0.1</probability></response>
        2. Simple format: ActionName[target]: probability
        3. Reverse format: probability: ActionName[target]

        Args:
            response: The LLM's response containing the distribution

        Returns:
            VerbalizedDistribution with parsed actions and probabilities
        """
        actions = []
        reasoning = ""

        # Extract reasoning (if present)
        reasoning_match = re.search(r"Reasoning:\s*(.+?)(?=\n\n|Distribution:|<response>|$)", response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        # Try XML format first: <response><action>...</action><probability>...</probability></response>
        xml_pattern = r"<response>\s*<action>(\w+)\[([^\]]+)\]</action>\s*<probability>(\d*\.?\d+)</probability>\s*</response>"
        xml_matches = re.findall(xml_pattern, response, re.IGNORECASE | re.DOTALL)

        if xml_matches:
            for match in xml_matches:
                action, target, prob_str = match
                try:
                    prob = float(prob_str)
                    if prob > 1.0:
                        prob /= 100.0
                    actions.append(ActionProbability(
                        action=action,
                        target=target,
                        probability=prob,
                    ))
                except ValueError:
                    continue
        else:
            # Fallback to original patterns
            for pattern in self.PROB_PATTERNS:
                matches = re.findall(pattern, response, re.IGNORECASE)
                for match in matches:
                    if len(match) == 3:
                        # Determine order based on pattern
                        if match[0].replace(".", "").isdigit():
                            # probability: Action[target]
                            prob_str, action, target = match
                        else:
                            # Action[target]: probability
                            action, target, prob_str = match

                        try:
                            prob = float(prob_str)
                            # Convert percentage to decimal if needed
                            if prob > 1.0:
                                prob /= 100.0

                            actions.append(ActionProbability(
                                action=action,
                                target=target,
                                probability=prob,
                            ))
                        except ValueError:
                            continue

        # Remove duplicates (keep first occurrence)
        seen = set()
        unique_actions = []
        for a in actions:
            key = (a.action, a.target)
            if key not in seen:
                seen.add(key)
                unique_actions.append(a)

        distribution = VerbalizedDistribution(
            actions=unique_actions,
            reasoning=reasoning,
            raw_response=response,
        )

        # Normalize probabilities
        if distribution.actions:
            distribution.normalize()

        return distribution

    def select_action(
        self,
        llm_client: Any,
        available_actions: List[Dict[str, Any]],
        world_description: str,
        exploration_history: Optional[List[Dict[str, Any]]] = None,
        agent_id: str = "agent_0",
        agent_rng: Optional[random.Random] = None,
    ) -> Tuple[str, Optional[str], str, VerbalizedDistribution]:
        """
        Use verbalized sampling to select an action.

        Args:
            llm_client: LLM client for generating the distribution
            available_actions: List of available actions
            world_description: Current world state
            exploration_history: Recent actions
            agent_id: Agent identifier
            agent_rng: Agent-specific RNG for reproducibility

        Returns:
            Tuple of (action_name, target, reasoning, distribution)
        """
        # Build the distribution prompt
        prompt = self.build_distribution_prompt(
            available_actions=available_actions,
            world_description=world_description,
            exploration_history=exploration_history,
            agent_id=agent_id,
        )

        # Generate distribution from LLM
        response = llm_client.generate(prompt, stop=None)

        # Parse the distribution
        distribution = self.parse_distribution(response)

        # Handle empty distribution (fallback to uniform over available actions)
        if not distribution.actions:
            print(f"[VerbalizedSampler] Warning: Could not parse distribution, using uniform fallback")
            fallback_actions = []
            for action in available_actions[:10]:
                name = action.get("name", "Navigate")
                targets = action.get("targets", [])
                if targets:
                    target = targets[0]
                    fallback_actions.append(ActionProbability(
                        action=name,
                        target=target,
                        probability=1.0 / min(len(available_actions), 10),
                    ))
            distribution.actions = fallback_actions
            distribution.normalize()

        # Sample from the distribution
        sampled = distribution.sample(rng=agent_rng)

        # Build reasoning that includes the distribution info
        top_actions = distribution.get_top_k(3)
        dist_summary = ", ".join(f"{a.full_action}:{a.probability:.2f}" for a in top_actions)
        reasoning = f"{distribution.reasoning} [VS Distribution: {dist_summary}] Sampled: {sampled.full_action}"

        return sampled.action, sampled.target, reasoning, distribution


def create_verbalized_sampler(temperature_boost: float = 0.0) -> VerbalizedSampler:
    """Factory function to create a VerbalizedSampler instance."""
    return VerbalizedSampler(temperature_boost=temperature_boost)
