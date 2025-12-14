"""
Trajectory analyzer for EMTOM task generation.

Analyzes exploration trajectories to identify patterns, surprises,
and opportunities for theory of mind tasks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DiscoveredMechanic:
    """A mechanic pattern discovered in a trajectory."""

    mechanic_type: str  # inverse, hidden_mapping, conditional, delayed, per_agent
    description: str
    evidence_steps: List[int]
    confidence: float = 0.8


@dataclass
class TaskOpportunity:
    """An opportunity for generating a task from the trajectory."""

    task_type: str  # theory_of_mind, planning, causal_reasoning, prediction
    description: str
    relevant_steps: List[int]
    key_entities: List[str]
    difficulty: int  # 1-5


@dataclass
class TrajectoryAnalysis:
    """Complete analysis of a trajectory."""

    episode_id: str
    discovered_mechanics: List[DiscoveredMechanic]
    mental_model_requirements: List[str]
    tom_challenges: List[str]
    task_opportunities: List[TaskOpportunity]
    key_surprise_moments: List[Dict[str, Any]]
    summary: str


class TrajectoryAnalyzer:
    """
    Analyzes trajectories to extract patterns for task generation.

    Can use either rule-based analysis or LLM-based analysis.
    """

    def __init__(self, llm_client: Any = None):
        """
        Initialize the analyzer.

        Args:
            llm_client: Optional LLM client for deeper analysis
        """
        self.llm = llm_client

    def analyze(self, trajectory: Dict[str, Any]) -> TrajectoryAnalysis:
        """
        Analyze a trajectory for task generation opportunities.

        Args:
            trajectory: Trajectory dict (from TrajectoryLogger)

        Returns:
            TrajectoryAnalysis with discovered patterns and task opportunities
        """
        episode_id = trajectory.get("episode_id", "unknown")
        steps = trajectory.get("steps", [])
        surprises = trajectory.get("surprise_summary", [])
        mechanics_active = trajectory.get("mechanics_active", [])

        # Discover mechanics from patterns
        discovered = self._discover_mechanics(steps, surprises, mechanics_active)

        # Identify mental model requirements
        mental_models = self._identify_mental_models(discovered, surprises)

        # Identify theory of mind challenges
        tom_challenges = self._identify_tom_challenges(steps, surprises)

        # Find task opportunities
        opportunities = self._find_task_opportunities(
            steps, surprises, discovered, tom_challenges
        )

        # Generate summary
        summary = self._generate_summary(
            trajectory, discovered, surprises, opportunities
        )

        return TrajectoryAnalysis(
            episode_id=episode_id,
            discovered_mechanics=discovered,
            mental_model_requirements=mental_models,
            tom_challenges=tom_challenges,
            task_opportunities=opportunities,
            key_surprise_moments=surprises,
            summary=summary,
        )

    def _discover_mechanics(
        self,
        steps: List[Dict],
        surprises: List[Dict],
        known_mechanics: List[str],
    ) -> List[DiscoveredMechanic]:
        """Discover mechanics from trajectory patterns."""
        discovered = []

        # Check for inverse effects
        inverse_evidence = self._find_inverse_patterns(steps, surprises)
        if inverse_evidence:
            discovered.append(DiscoveredMechanic(
                mechanic_type="inverse",
                description="Actions have opposite effects from expected",
                evidence_steps=inverse_evidence,
                confidence=0.9,
            ))

        # Check for hidden mappings
        mapping_evidence = self._find_mapping_patterns(steps, surprises)
        if mapping_evidence:
            discovered.append(DiscoveredMechanic(
                mechanic_type="hidden_mapping",
                description="Controls affect objects in different locations",
                evidence_steps=mapping_evidence,
                confidence=0.8,
            ))

        # Check for conditional/counting patterns
        conditional_evidence = self._find_conditional_patterns(steps, surprises)
        if conditional_evidence:
            discovered.append(DiscoveredMechanic(
                mechanic_type="conditional",
                description="Actions require multiple attempts or specific conditions",
                evidence_steps=conditional_evidence,
                confidence=0.85,
            ))

        # Check for delayed effects
        delayed_evidence = self._find_delayed_patterns(steps)
        if delayed_evidence:
            discovered.append(DiscoveredMechanic(
                mechanic_type="delayed",
                description="Effects occur after a delay",
                evidence_steps=delayed_evidence,
                confidence=0.7,
            ))

        return discovered

    def _find_inverse_patterns(
        self, steps: List[Dict], surprises: List[Dict]
    ) -> List[int]:
        """Find steps where inverse effects occurred."""
        evidence = []
        for surprise in surprises:
            explanation = surprise.get("explanation", "").lower()
            if "inverse" in explanation or "opposite" in explanation:
                evidence.append(surprise.get("step", 0))

        # Also check step observations for inverse keywords
        for step in steps:
            obs = str(step.get("observations", {})).lower()
            if "instead" in obs or "opposite" in obs:
                if step["step"] not in evidence:
                    evidence.append(step["step"])

        return evidence

    def _find_mapping_patterns(
        self, steps: List[Dict], surprises: List[Dict]
    ) -> List[int]:
        """Find steps where remote/hidden mappings were observed."""
        evidence = []
        for surprise in surprises:
            explanation = surprise.get("explanation", "").lower()
            if (
                "remote" in explanation
                or "elsewhere" in explanation
                or "different location" in explanation
                or "no immediate" in explanation
            ):
                evidence.append(surprise.get("step", 0))

        return evidence

    def _find_conditional_patterns(
        self, steps: List[Dict], surprises: List[Dict]
    ) -> List[int]:
        """Find steps where conditional/counting mechanics were observed."""
        evidence = []

        # Look for repeated actions on same target
        action_counts: Dict[str, List[int]] = {}
        for step in steps:
            for agent_id, action_info in step.get("agent_actions", {}).items():
                action = action_info.get("action", "")
                target = action_info.get("target", "")
                if action and target:
                    key = f"{action}:{target}"
                    if key not in action_counts:
                        action_counts[key] = []
                    action_counts[key].append(step["step"])

        # Check surprises for "nothing happened" patterns
        for surprise in surprises:
            explanation = surprise.get("explanation", "").lower()
            if "nothing happened" in explanation or "no effect" in explanation:
                evidence.append(surprise.get("step", 0))

        return evidence

    def _find_delayed_patterns(self, steps: List[Dict]) -> List[int]:
        """Find steps where delayed effects occurred."""
        evidence = []
        for step in steps:
            for effect in step.get("effects", []):
                if effect.get("was_delayed", False):
                    evidence.append(step["step"])
        return evidence

    def _identify_mental_models(
        self,
        discovered: List[DiscoveredMechanic],
        surprises: List[Dict],
    ) -> List[str]:
        """Identify mental models needed to understand the world."""
        models = []

        for mechanic in discovered:
            if mechanic.mechanic_type == "inverse":
                models.append(
                    "Understanding that some actions have opposite effects"
                )
            elif mechanic.mechanic_type == "hidden_mapping":
                models.append(
                    "Mapping controls to their remote effects"
                )
            elif mechanic.mechanic_type == "conditional":
                models.append(
                    "Understanding activation conditions (e.g., repeated attempts)"
                )
            elif mechanic.mechanic_type == "delayed":
                models.append(
                    "Tracking delayed effects and their timing"
                )

        # Add models from surprise hypotheses
        for surprise in surprises:
            hypothesis = surprise.get("hypothesis", "")
            if hypothesis and hypothesis not in models:
                models.append(f"Hypothesis: {hypothesis}")

        return models

    def _identify_tom_challenges(
        self, steps: List[Dict], surprises: List[Dict]
    ) -> List[str]:
        """Identify theory of mind challenges in the trajectory."""
        challenges = []

        # Multi-agent scenarios
        agents = set()
        for step in steps:
            agents.update(step.get("agent_actions", {}).keys())

        if len(agents) > 1:
            challenges.append(
                "Multiple agents may have different knowledge about mechanics"
            )
            challenges.append(
                "Agents need to communicate discoveries to coordinate"
            )

        # Hidden information challenges
        for surprise in surprises:
            if surprise.get("surprise_level", 0) >= 4:
                challenges.append(
                    f"High-surprise discovery at step {surprise['step']} "
                    f"reveals hidden world behavior"
                )

        return challenges

    def _find_task_opportunities(
        self,
        steps: List[Dict],
        surprises: List[Dict],
        discovered: List[DiscoveredMechanic],
        challenges: List[str],
    ) -> List[TaskOpportunity]:
        """Find opportunities for generating tasks."""
        opportunities = []

        # Theory of Mind tasks from surprises
        for surprise in surprises:
            if surprise.get("surprise_level", 0) >= 3:
                opportunities.append(TaskOpportunity(
                    task_type="theory_of_mind",
                    description=(
                        f"Predict what another agent would expect when "
                        f"performing '{surprise.get('action')}' on '{surprise.get('target')}'"
                    ),
                    relevant_steps=[surprise.get("step", 0)],
                    key_entities=[surprise.get("target", "unknown")],
                    difficulty=surprise.get("surprise_level", 3),
                ))

        # Causal reasoning tasks from mechanics
        for mechanic in discovered:
            if mechanic.mechanic_type == "hidden_mapping":
                opportunities.append(TaskOpportunity(
                    task_type="causal_reasoning",
                    description="Determine which control affects which object",
                    relevant_steps=mechanic.evidence_steps,
                    key_entities=[],  # Would need to extract from steps
                    difficulty=4,
                ))
            elif mechanic.mechanic_type == "conditional":
                opportunities.append(TaskOpportunity(
                    task_type="causal_reasoning",
                    description="Discover the activation condition for an object",
                    relevant_steps=mechanic.evidence_steps,
                    key_entities=[],
                    difficulty=3,
                ))

        # Planning tasks
        if discovered:
            opportunities.append(TaskOpportunity(
                task_type="planning",
                description=(
                    "Plan a sequence of actions to achieve a goal "
                    "given the discovered mechanics"
                ),
                relevant_steps=[s["step"] for s in steps[:5]],
                key_entities=[],
                difficulty=4,
            ))

        # Prediction tasks
        for mechanic in discovered:
            opportunities.append(TaskOpportunity(
                task_type="prediction",
                description=(
                    f"Predict the outcome of an action given "
                    f"the {mechanic.mechanic_type} mechanic"
                ),
                relevant_steps=mechanic.evidence_steps[:3],
                key_entities=[],
                difficulty=3,
            ))

        return opportunities

    def _generate_summary(
        self,
        trajectory: Dict,
        discovered: List[DiscoveredMechanic],
        surprises: List[Dict],
        opportunities: List[TaskOpportunity],
    ) -> str:
        """Generate a human-readable summary of the analysis."""
        lines = []

        stats = trajectory.get("statistics", {})
        lines.append(f"Trajectory Analysis Summary")
        lines.append(f"=" * 40)
        lines.append(f"Episode: {trajectory.get('episode_id', 'unknown')}")
        lines.append(f"Total steps: {stats.get('total_steps', 0)}")
        lines.append(f"Total surprises: {stats.get('total_surprises', 0)}")
        lines.append("")

        if discovered:
            lines.append("Discovered Mechanics:")
            for m in discovered:
                lines.append(f"  - {m.mechanic_type}: {m.description}")
                lines.append(f"    Evidence at steps: {m.evidence_steps}")
        else:
            lines.append("No mechanics discovered")
        lines.append("")

        if opportunities:
            lines.append(f"Task Opportunities: {len(opportunities)}")
            for opp in opportunities[:5]:  # Show first 5
                lines.append(f"  - [{opp.task_type}] {opp.description}")
        lines.append("")

        return "\n".join(lines)

    def analyze_with_llm(self, trajectory: Dict[str, Any]) -> TrajectoryAnalysis:
        """
        Analyze trajectory using LLM for deeper insights.

        Falls back to rule-based if no LLM available.
        """
        if not self.llm:
            return self.analyze(trajectory)

        # First do rule-based analysis
        base_analysis = self.analyze(trajectory)

        # Then enhance with LLM
        prompt = self._build_llm_prompt(trajectory, base_analysis)
        response = self.llm.generate(prompt)

        # Parse LLM response and merge with base analysis
        llm_insights = self._parse_llm_analysis(response)

        # Merge insights
        base_analysis.tom_challenges.extend(
            llm_insights.get("tom_challenges", [])
        )
        base_analysis.task_opportunities.extend(
            llm_insights.get("task_opportunities", [])
        )

        return base_analysis

    def _build_llm_prompt(
        self, trajectory: Dict, base_analysis: TrajectoryAnalysis
    ) -> str:
        """Build prompt for LLM analysis."""
        return f"""Analyze this exploration trajectory for theory of mind benchmark task generation.

Trajectory summary:
- Total steps: {trajectory.get('statistics', {}).get('total_steps', 0)}
- Surprise moments: {trajectory.get('statistics', {}).get('total_surprises', 0)}
- Mechanics discovered: {[m.mechanic_type for m in base_analysis.discovered_mechanics]}

Key surprise moments:
{json.dumps(trajectory.get('surprise_summary', [])[:5], indent=2)}

Based on this trajectory, identify:
1. What theory of mind reasoning challenges exist?
2. What additional task types could be generated?
3. What mental models would an agent need?

Respond in JSON:
{{
    "tom_challenges": ["..."],
    "task_opportunities": [
        {{"task_type": "...", "description": "...", "difficulty": 1-5}}
    ],
    "mental_models": ["..."]
}}"""

    def _parse_llm_analysis(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into analysis components."""
        try:
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
        return {}
