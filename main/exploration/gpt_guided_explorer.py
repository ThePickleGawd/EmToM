#!/usr/bin/env python3
"""
GPT-Guided Room Explorer Module.

This module uses OpenAI's GPT to guide robot exploration through rooms,
making decisions about which rooms to visit, what objects to interact with,
and forming hypotheses about surprising findings.

Usage:
    python -m main.exploration.gpt_guided_explorer \
        hydra.run.dir="." \
        +skill_runner_episode_id="334" \
        +live_display=True
"""

import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from dotenv import load_dotenv

from habitat_llm.utils import cprint


# Load environment variables from .env file
load_dotenv()


@dataclass
class Interaction:
    """Record of a single interaction with an object."""
    action: str
    target: str
    result: str
    hypothesis: str
    surprising: bool
    finding: Optional[str] = None


@dataclass
class RoomVisit:
    """Record of visiting a single room."""
    room_name: str
    furniture_observed: List[str] = field(default_factory=list)
    objects_observed: List[str] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)
    gpt_observations: str = ""
    gpt_hypotheses: List[str] = field(default_factory=list)
    surprising_findings: List[str] = field(default_factory=list)


@dataclass
class ExplorationSession:
    """Complete exploration session data."""
    start_time: str
    end_time: Optional[str] = None
    rooms_visited: List[RoomVisit] = field(default_factory=list)
    total_interactions: int = 0
    total_surprising_findings: int = 0
    gpt_model: str = "gpt-4o-mini"


class GPTExplorationGuide:
    """
    GPT-based guide for making exploration decisions.

    Uses OpenAI's API to decide which rooms to explore and what
    interactions to perform, while generating hypotheses about findings.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize the GPT guide.

        Args:
            model: OpenAI model to use (default: gpt-4o-mini for cost efficiency)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please set it in your .env file."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize with system prompt
        self.system_prompt = """You are an AI assistant guiding a robot's exploration of a household environment.
Your goal is to help the robot explore rooms systematically, interact with interesting objects,
and identify anything surprising or noteworthy.

When choosing rooms, consider:
- Diversity: try to explore different types of spaces
- Interest: prioritize rooms that might have interactive elements
- Never suggest a room that has already been visited

When suggesting interactions:
- Pick up small, portable objects to examine them
- Open containers, cabinets, and appliances to see contents
- Press buttons or switches when you find them
- Place objects back or in new locations after examining

When forming hypotheses:
- Consider what you would expect to find in each room
- Note anything that seems out of place or unexpected
- Think about the purpose of objects and their locations

Always be curious and observant. Report findings in a clear, scientific manner."""

    def _chat(self, user_message: str) -> str:
        """Send a message to GPT and get a response."""
        self.conversation_history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def choose_next_room(
        self,
        available_rooms: List[str],
        visited_rooms: List[str]
    ) -> Tuple[str, str]:
        """
        Ask GPT to choose the next room to explore.

        Args:
            available_rooms: List of all room names in the scene
            visited_rooms: List of already visited room names

        Returns:
            Tuple of (chosen_room_name, reasoning)
        """
        unvisited = [r for r in available_rooms if r not in visited_rooms]

        if not unvisited:
            return None, "All rooms have been visited"

        prompt = f"""Choose the next room to explore.

Available rooms (not yet visited): {unvisited}
Already visited: {visited_rooms if visited_rooms else "None yet"}

Respond in this exact format:
ROOM: <room_name>
REASONING: <one sentence explaining your choice>"""

        response = self._chat(prompt)

        # Parse response
        lines = response.strip().split('\n')
        chosen_room = None
        reasoning = ""

        for line in lines:
            if line.startswith("ROOM:"):
                chosen_room = line.replace("ROOM:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning = line.replace("REASONING:", "").strip()

        # Validate the room choice
        if chosen_room not in unvisited:
            # GPT made an invalid choice, pick randomly
            chosen_room = random.choice(unvisited)
            reasoning = f"(Random fallback - GPT suggested invalid room) {reasoning}"

        return chosen_room, reasoning

    def analyze_room(
        self,
        room_name: str,
        furniture_list: List[str],
        objects_list: List[str]
    ) -> Dict[str, Any]:
        """
        Ask GPT to analyze a room and suggest interactions.

        Args:
            room_name: Name of the current room
            furniture_list: List of furniture in the room
            objects_list: List of objects in/on furniture

        Returns:
            Dict with observations, suggested_interactions, and initial_hypotheses
        """
        prompt = f"""You have entered the {room_name}.

Furniture in this room: {furniture_list if furniture_list else "None visible"}
Objects visible: {objects_list if objects_list else "None visible"}

Analyze this room and suggest what to do. Respond in this format:

OBSERVATIONS: <describe what you see and your initial impressions>
HYPOTHESES: <list 1-3 hypotheses about this room or its contents, separated by |>
INTERACTIONS: <list up to 3 suggested interactions in format ACTION:TARGET, separated by |>

Valid actions are: Pick, Place, Open, Close, Navigate
Example interactions: Pick:apple_0|Open:fridge_0|Navigate:kitchen_counter_0"""

        response = self._chat(prompt)

        result = {
            "observations": "",
            "hypotheses": [],
            "interactions": []
        }

        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("OBSERVATIONS:"):
                result["observations"] = line.replace("OBSERVATIONS:", "").strip()
            elif line.startswith("HYPOTHESES:"):
                hypotheses_str = line.replace("HYPOTHESES:", "").strip()
                result["hypotheses"] = [h.strip() for h in hypotheses_str.split('|') if h.strip()]
            elif line.startswith("INTERACTIONS:"):
                interactions_str = line.replace("INTERACTIONS:", "").strip()
                for interaction in interactions_str.split('|'):
                    interaction = interaction.strip()
                    if ':' in interaction:
                        action, target = interaction.split(':', 1)
                        result["interactions"].append({
                            "action": action.strip(),
                            "target": target.strip()
                        })

        return result

    def evaluate_interaction_result(
        self,
        action: str,
        target: str,
        result: str,
        room_context: str
    ) -> Dict[str, Any]:
        """
        Ask GPT to evaluate the result of an interaction.

        Args:
            action: The action that was performed
            target: The target of the action
            result: The result/response from the action
            room_context: Context about the current room

        Returns:
            Dict with hypothesis, surprising (bool), and finding (if surprising)
        """
        prompt = f"""Evaluate the result of this interaction:

Room: {room_context}
Action: {action} on {target}
Result: {result}

Respond in this format:
HYPOTHESIS: <what this result tells us or confirms>
SURPRISING: <YES or NO - is this unexpected or noteworthy?>
FINDING: <if surprising, explain what makes it interesting; otherwise leave blank>"""

        response = self._chat(prompt)

        evaluation = {
            "hypothesis": "",
            "surprising": False,
            "finding": None
        }

        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("HYPOTHESIS:"):
                evaluation["hypothesis"] = line.replace("HYPOTHESIS:", "").strip()
            elif line.startswith("SURPRISING:"):
                surprising_str = line.replace("SURPRISING:", "").strip().upper()
                evaluation["surprising"] = surprising_str.startswith("YES")
            elif line.startswith("FINDING:"):
                finding = line.replace("FINDING:", "").strip()
                if finding and finding.lower() not in ["", "n/a", "none", "blank"]:
                    evaluation["finding"] = finding

        return evaluation

    def summarize_exploration(
        self,
        session: ExplorationSession
    ) -> str:
        """
        Ask GPT to summarize the entire exploration session.

        Args:
            session: The complete exploration session data

        Returns:
            Summary string
        """
        rooms_summary = []
        for visit in session.rooms_visited:
            rooms_summary.append(f"- {visit.room_name}: {len(visit.interactions)} interactions, "
                               f"{len(visit.surprising_findings)} surprising findings")

        prompt = f"""Summarize this exploration session:

Rooms visited:
{chr(10).join(rooms_summary)}

Total interactions: {session.total_interactions}
Total surprising findings: {session.total_surprising_findings}

All surprising findings:
{chr(10).join([f"- {visit.room_name}: {finding}"
               for visit in session.rooms_visited
               for finding in visit.surprising_findings]) or "None"}

Provide a brief (2-3 sentence) summary of what was discovered during this exploration."""

        return self._chat(prompt)


class GPTGuidedExplorer:
    """
    Main class for GPT-guided exploration.

    Integrates GPT decision-making with the robot's exploration capabilities.
    """

    def __init__(
        self,
        room_explorer: Any,
        gpt_model: str = "gpt-4o-mini",
        output_dir: str = "./exploration_results"
    ):
        """
        Initialize the GPT-guided explorer.

        Args:
            room_explorer: Instance of RoomExplorer from room_explorer.py
            gpt_model: OpenAI model to use
            output_dir: Directory to save JSON results
        """
        self.room_explorer = room_explorer
        self.gpt_guide = GPTExplorationGuide(model=gpt_model)
        self.output_dir = output_dir
        self.session: Optional[ExplorationSession] = None

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def _get_objects_near_furniture(self, furniture_name: str) -> List[str]:
        """Get objects that might be on or near a piece of furniture."""
        all_objects = self.room_explorer.world_graph.get_all_objects()
        nearby_objects = []

        # Get furniture node
        furniture_node = self.room_explorer.world_graph.get_node_from_name(furniture_name)
        if furniture_node is None:
            return []

        # Find objects associated with this furniture
        for obj in all_objects:
            try:
                parent_furniture = self.room_explorer.world_graph.find_furniture_for_object(obj)
                if parent_furniture and parent_furniture.name == furniture_name:
                    nearby_objects.append(obj.name)
            except Exception:
                continue

        return nearby_objects

    def _get_all_objects_in_room(self, room_name: str) -> List[str]:
        """Get all objects in a room."""
        furniture_list = self.room_explorer.get_furniture_in_room(room_name)
        all_objects = []

        for furniture in furniture_list:
            objects = self._get_objects_near_furniture(furniture.name)
            all_objects.extend(objects)

        return all_objects

    def _execute_interaction(
        self,
        action: str,
        target: str,
        room_name: str
    ) -> Tuple[str, bool]:
        """
        Execute an interaction with the environment.

        Args:
            action: The action to perform (Pick, Place, Open, Close, Navigate)
            target: The target object/furniture
            room_name: Current room for context

        Returns:
            Tuple of (result_message, success)
        """
        # Map action to skill
        valid_actions = ["Pick", "Place", "Open", "Close", "Navigate"]

        if action not in valid_actions:
            return f"Invalid action: {action}", False

        try:
            high_level_skill_actions = {
                self.room_explorer.robot_agent.uid: (action, target, None)
            }

            if self.room_explorer.live_display_enabled:
                responses, frames = self.room_explorer._execute_skill_with_live_display(
                    high_level_skill_actions,
                    room_name=room_name,
                    furniture_name=target,
                )
            else:
                from habitat_llm.examples.example_utils import execute_skill

                responses, _, frames = execute_skill(
                    high_level_skill_actions,
                    self.room_explorer.planner,
                    make_video=False,
                    play_video=False,
                )

            response_msg = responses[self.room_explorer.robot_agent.uid]
            success = "success" in response_msg.lower() or "successful" in response_msg.lower()
            return response_msg, success

        except Exception as e:
            return f"Interaction failed: {str(e)}", False

    def _print_and_log(self, message: str, color: str = None):
        """Print message to terminal with color.

        Supported colors: red, green, blue, gray, yellow, None
        """
        # Map unsupported colors to supported ones
        color_map = {
            "cyan": "blue",
            "white": None,
        }
        mapped_color = color_map.get(color, color)
        cprint(message, mapped_color)

    def explore_with_gpt(
        self,
        max_rooms: Optional[int] = None,
        max_interactions_per_room: int = 3
    ) -> ExplorationSession:
        """
        Run GPT-guided exploration of all rooms.

        Args:
            max_rooms: Maximum rooms to visit (None = all rooms)
            max_interactions_per_room: Max interactions per room

        Returns:
            ExplorationSession with all data
        """
        self.session = ExplorationSession(
            start_time=datetime.now().isoformat(),
            gpt_model=self.gpt_guide.model
        )

        all_rooms = self.room_explorer.get_room_names()
        visited_rooms: List[str] = []

        self._print_and_log("\n" + "=" * 60, "green")
        self._print_and_log("🤖 GPT-GUIDED EXPLORATION STARTING", "green")
        self._print_and_log(f"   Model: {self.gpt_guide.model}", "blue")
        self._print_and_log(f"   Available rooms: {len(all_rooms)}", "blue")
        self._print_and_log("=" * 60 + "\n", "green")

        rooms_to_visit = max_rooms if max_rooms else len(all_rooms)

        while len(visited_rooms) < rooms_to_visit and len(visited_rooms) < len(all_rooms):
            # Ask GPT to choose next room
            self._print_and_log("\n[GPT] Choosing next room to explore...", "yellow")

            chosen_room, reasoning = self.gpt_guide.choose_next_room(all_rooms, visited_rooms)

            if chosen_room is None:
                self._print_and_log("All rooms visited!", "green")
                break

            self._print_and_log(f"[GPT] Selected: {chosen_room}", "yellow")
            self._print_and_log(f"      Reasoning: {reasoning}", "yellow")

            # Create room visit record
            room_visit = RoomVisit(room_name=chosen_room)

            # Get room contents
            furniture_list = self.room_explorer.get_furniture_in_room(chosen_room)
            furniture_names = [f.name for f in furniture_list]
            objects_in_room = self._get_all_objects_in_room(chosen_room)

            room_visit.furniture_observed = furniture_names
            room_visit.objects_observed = objects_in_room

            self._print_and_log(f"\n>>> ENTERING: {chosen_room}", "green")
            self._print_and_log(f"    Furniture: {furniture_names}", "blue")
            self._print_and_log(f"    Objects: {objects_in_room}", "blue")

            # Navigate to first furniture in room
            if furniture_names:
                self._print_and_log(f"\n[Robot] Navigating to {furniture_names[0]}...", "cyan")
                nav_result, nav_frames, nav_success = self.room_explorer.navigate_to_furniture(
                    furniture_names[0],
                    room_name=chosen_room,
                    make_video=False
                )
                self._print_and_log(f"        Result: {nav_result}", "cyan")

            # Ask GPT to analyze room and suggest interactions
            self._print_and_log("\n[GPT] Analyzing room...", "yellow")

            analysis = self.gpt_guide.analyze_room(
                chosen_room,
                furniture_names,
                objects_in_room
            )

            room_visit.gpt_observations = analysis["observations"]
            room_visit.gpt_hypotheses = analysis["hypotheses"]

            self._print_and_log(f"[GPT] Observations: {analysis['observations']}", "yellow")
            self._print_and_log(f"[GPT] Hypotheses:", "yellow")
            for h in analysis["hypotheses"]:
                self._print_and_log(f"      - {h}", "yellow")

            # Execute suggested interactions
            interactions_done = 0
            for interaction in analysis["interactions"][:max_interactions_per_room]:
                action = interaction["action"]
                target = interaction["target"]

                self._print_and_log(f"\n[Robot] Attempting: {action} on {target}...", "cyan")

                result_msg, success = self._execute_interaction(action, target, chosen_room)

                self._print_and_log(f"        Result: {result_msg}", "cyan" if success else "red")

                # Ask GPT to evaluate
                self._print_and_log("[GPT] Evaluating result...", "yellow")

                evaluation = self.gpt_guide.evaluate_interaction_result(
                    action, target, result_msg,
                    f"{chosen_room} with {furniture_names}"
                )

                # Record interaction
                interaction_record = Interaction(
                    action=action,
                    target=target,
                    result=result_msg,
                    hypothesis=evaluation["hypothesis"],
                    surprising=evaluation["surprising"],
                    finding=evaluation["finding"]
                )
                room_visit.interactions.append(interaction_record)

                self._print_and_log(f"[GPT] Hypothesis: {evaluation['hypothesis']}", "yellow")

                if evaluation["surprising"]:
                    self._print_and_log(f"[GPT] ⚠️  SURPRISING FINDING: {evaluation['finding']}", "red")
                    room_visit.surprising_findings.append(evaluation["finding"])
                    self.session.total_surprising_findings += 1

                interactions_done += 1
                self.session.total_interactions += 1

            # Mark room as visited
            visited_rooms.append(chosen_room)
            self.session.rooms_visited.append(room_visit)

            self._print_and_log(f"\n<<< FINISHED: {chosen_room}", "green")
            self._print_and_log(f"    Interactions: {interactions_done}", "blue")
            self._print_and_log(f"    Surprising findings: {len(room_visit.surprising_findings)}", "blue")

        # Finalize session
        self.session.end_time = datetime.now().isoformat()

        # Get GPT summary
        self._print_and_log("\n" + "=" * 60, "green")
        self._print_and_log("[GPT] Generating exploration summary...", "yellow")

        summary = self.gpt_guide.summarize_exploration(self.session)
        self._print_and_log(f"\n{summary}", "yellow")

        # Save to JSON
        json_path = self._save_session_to_json()

        self._print_and_log("\n" + "=" * 60, "green")
        self._print_and_log("🤖 GPT-GUIDED EXPLORATION COMPLETE", "green")
        self._print_and_log(f"   Rooms visited: {len(visited_rooms)}", "blue")
        self._print_and_log(f"   Total interactions: {self.session.total_interactions}", "blue")
        self._print_and_log(f"   Surprising findings: {self.session.total_surprising_findings}", "blue")
        self._print_and_log(f"   Results saved to: {json_path}", "blue")
        self._print_and_log("=" * 60 + "\n", "green")

        return self.session

    def _save_session_to_json(self) -> str:
        """Save the exploration session to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gpt_exploration_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Convert dataclasses to dict
        session_dict = {
            "start_time": self.session.start_time,
            "end_time": self.session.end_time,
            "gpt_model": self.session.gpt_model,
            "total_interactions": self.session.total_interactions,
            "total_surprising_findings": self.session.total_surprising_findings,
            "rooms_visited": []
        }

        for visit in self.session.rooms_visited:
            visit_dict = {
                "room_name": visit.room_name,
                "furniture_observed": visit.furniture_observed,
                "objects_observed": visit.objects_observed,
                "gpt_observations": visit.gpt_observations,
                "gpt_hypotheses": visit.gpt_hypotheses,
                "surprising_findings": visit.surprising_findings,
                "interactions": []
            }

            for interaction in visit.interactions:
                interaction_dict = {
                    "action": interaction.action,
                    "target": interaction.target,
                    "result": interaction.result,
                    "hypothesis": interaction.hypothesis,
                    "surprising": interaction.surprising,
                    "finding": interaction.finding
                }
                visit_dict["interactions"].append(interaction_dict)

            session_dict["rooms_visited"].append(visit_dict)

        with open(filepath, 'w') as f:
            json.dump(session_dict, f, indent=2)

        return filepath
