"""
Room Explorer module for systematically exploring all rooms in a Habitat scene.

This module provides the RoomExplorer class which can be used to navigate a robot
through all rooms in a scene, visiting furniture within each room.
"""

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from habitat_llm.utils import cprint


@dataclass
class RoomExplorationResult:
    """Results from exploring a single room."""

    room_name: str
    furniture_visited: List[str] = field(default_factory=list)
    navigation_results: List[Tuple[str, str, bool]] = field(default_factory=list)
    frames: List[Any] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class FullExplorationResult:
    """Results from exploring all rooms in a scene."""

    rooms_explored: List[RoomExplorationResult] = field(default_factory=list)
    total_rooms: int = 0
    total_furniture_visited: int = 0
    all_frames: List[Any] = field(default_factory=list)


class LiveDisplay:
    """
    Real-time X11 display for showing robot's view during exploration.

    The display window stays open throughout the exploration and updates
    in real-time as the robot moves through the scene.
    """

    def __init__(self, window_name: str = "Robot Exploration", scale: float = 1.0):
        """
        Initialize the live display.

        Args:
            window_name: Name of the X11 window.
            scale: Scale factor for the display (1.0 = original size).
        """
        self.window_name = window_name
        self.scale = scale
        self._window_created = False
        self._current_action = ""
        self._current_room = ""

    def _ensure_window(self) -> None:
        """Create the window if it doesn't exist."""
        if not self._window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

    def set_status(self, room: str = "", action: str = "") -> None:
        """Update the status text shown on the display."""
        self._current_room = room
        self._current_action = action

    def show_frame(self, frame: np.ndarray) -> bool:
        """
        Display a frame in the X11 window.

        Args:
            frame: The frame to display (numpy array, RGB or BGR).

        Returns:
            False if user pressed 'q' to quit, True otherwise.
        """
        self._ensure_window()

        # Convert to contiguous array if needed
        display_frame = np.ascontiguousarray(frame)

        # Add status text overlay
        if self._current_room or self._current_action:
            # Add semi-transparent background for text
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0)

            # Add room name
            if self._current_room:
                cv2.putText(
                    display_frame,
                    f"Room: {self._current_room}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            # Add action
            if self._current_action:
                cv2.putText(
                    display_frame,
                    f"Action: {self._current_action}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    2,
                )

        # Scale if needed
        if self.scale != 1.0:
            new_width = int(display_frame.shape[1] * self.scale)
            new_height = int(display_frame.shape[0] * self.scale)
            display_frame = cv2.resize(display_frame, (new_width, new_height))

        # Convert RGB to BGR for OpenCV display
        if len(display_frame.shape) == 3 and display_frame.shape[2] == 3:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow(self.window_name, display_frame)

        # Check for quit key (non-blocking)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        return True

    def close(self) -> None:
        """Close the display window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class RoomExplorer:
    """
    A class for systematically exploring all rooms in a Habitat scene.

    This class navigates a robot agent through all rooms in the scene,
    visiting furniture within each room to thoroughly explore the environment.

    Usage:
        explorer = RoomExplorer(env_interface, planner, robot_agent, live_display=True)
        results = explorer.explore_all_rooms(furniture_per_room=3)
    """

    def __init__(
        self,
        env_interface: Any,
        planner: Any,
        robot_agent: Any,
        show_videos: bool = False,
        live_display: bool = False,
        display_scale: float = 1.0,
    ):
        """
        Initialize the RoomExplorer.

        Args:
            env_interface: The Habitat EnvironmentInterface instance.
            planner: The planner instance with initialized agents.
            robot_agent: The robot agent to control.
            show_videos: Whether to display videos after each navigation (legacy).
            live_display: Whether to show real-time X11 display during exploration.
            display_scale: Scale factor for the live display window.
        """
        self.env_interface = env_interface
        self.planner = planner
        self.robot_agent = robot_agent
        self.show_videos = show_videos
        self.live_display_enabled = live_display
        self.world_graph = env_interface.world_graph[robot_agent.uid]

        # Initialize live display if enabled
        self._live_display: Optional[LiveDisplay] = None
        if live_display:
            self._live_display = LiveDisplay(
                window_name="Robot Exploration",
                scale=display_scale,
            )

        # Track number of agents for frame extraction
        self._num_agents = 0
        for _agent_conf in env_interface.conf.evaluation.agents.values():
            self._num_agents += 1

    def _get_combined_frame(self, observations: Dict[str, Any]) -> np.ndarray:
        """Extract and combine third-person RGB frames from observations."""
        images = []
        for obs_name, obs_value in observations.items():
            if "third_rgb" in obs_name:
                if self._num_agents == 1:
                    if "0" in obs_name or "main_agent" in obs_name:
                        images.append(obs_value)
                else:
                    images.append(obs_value)

        if not images:
            # Fallback to any RGB observation
            for obs_name, obs_value in observations.items():
                if "rgb" in obs_name.lower():
                    images.append(obs_value)
                    break

        if not images:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        # Get the first image and convert to numpy
        img = images[0]
        if hasattr(img, "cpu"):
            img = img.cpu().numpy()
        if hasattr(img, "numpy"):
            img = img.numpy()

        # Handle batch dimension
        if img.ndim == 4:
            img = img[0]

        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        return img

    def _execute_skill_with_live_display(
        self,
        high_level_skill_actions: Dict[Any, Any],
        room_name: str = "",
        furniture_name: str = "",
    ) -> Tuple[Dict[Any, Any], List[Any]]:
        """
        Execute a skill while showing real-time display via X11.

        Args:
            high_level_skill_actions: Map of agent indices to actions.
            room_name: Current room name for display.
            furniture_name: Target furniture name for display.

        Returns:
            Tuple of (responses dict, list of frames).
        """
        frames = []
        observations = self.env_interface.get_observations()
        agent_idx = list(high_level_skill_actions.keys())[0]
        skill_name = high_level_skill_actions[agent_idx][0]

        # Update display status
        if self._live_display:
            self._live_display.set_status(
                room=room_name,
                action=f"{skill_name} -> {furniture_name}",
            )

        skill_steps = 0
        max_skill_steps = 1500
        skill_done = None
        user_quit = False

        while not skill_done and not user_quit:
            assert skill_steps < max_skill_steps, f"Maximum steps reached: {skill_name} fails."

            # Get low level actions and responses
            low_level_actions, responses = self.planner.process_high_level_actions(
                high_level_skill_actions, observations
            )

            assert len(low_level_actions) > 0, f"No low level actions. Response: {responses.values()}"

            # Check if agent finishes
            if any(responses.values()):
                skill_done = True

            # Step the environment
            obs, reward, done, info = self.env_interface.step(low_level_actions)
            observations = self.env_interface.parse_observations(obs)

            # Get frame and display it
            frame = self._get_combined_frame(observations)
            frames.append(frame)

            if self._live_display:
                if not self._live_display.show_frame(frame):
                    user_quit = True
                    cprint("\nUser requested quit (pressed 'q')", "yellow")

            skill_steps += 1

        return responses, frames

    def get_all_rooms(self) -> List[Any]:
        """Get all rooms in the scene."""
        return self.world_graph.get_all_rooms()

    def get_room_names(self) -> List[str]:
        """Get names of all rooms in the scene."""
        return [room.name for room in self.get_all_rooms()]

    def get_furniture_in_room(self, room_name: str) -> List[Any]:
        """Get all furniture in a specific room."""
        return self.world_graph.get_furniture_in_room(room_name)

    def navigate_to_furniture(
        self,
        furniture_name: str,
        room_name: str = "",
        make_video: bool = False,
        vid_postfix: str = "",
    ) -> Tuple[str, List[Any], bool]:
        """
        Navigate the robot to a specific piece of furniture.

        Args:
            furniture_name: Name of the furniture to navigate to.
            room_name: Current room name (for display).
            make_video: Whether to save video (ignored if live_display is enabled).
            vid_postfix: Postfix for video filename.

        Returns:
            Tuple of (response_message, frames, success).
        """
        high_level_skill_actions = {
            self.robot_agent.uid: ("Navigate", furniture_name, None)
        }

        try:
            if self.live_display_enabled:
                # Use live display mode
                responses, frames = self._execute_skill_with_live_display(
                    high_level_skill_actions,
                    room_name=room_name,
                    furniture_name=furniture_name,
                )
            else:
                # Use legacy video mode
                from habitat_llm.examples.example_utils import execute_skill

                responses, _, frames = execute_skill(
                    high_level_skill_actions,
                    self.planner,
                    vid_postfix=vid_postfix,
                    make_video=make_video,
                    play_video=self.show_videos,
                )

            response_msg = responses[self.robot_agent.uid]
            success = "success" in response_msg.lower() or "successful" in response_msg.lower()
            return response_msg, frames, success

        except Exception as e:
            return f"Navigation failed: {str(e)}", [], False

    def explore_room(
        self,
        room_name: str,
        furniture_limit: Optional[int] = None,
        randomize_furniture: bool = True,
        make_video: bool = False,
        on_furniture_visit: Optional[Callable[[str, str, str], None]] = None,
    ) -> RoomExplorationResult:
        """
        Explore a single room by visiting furniture within it.

        Args:
            room_name: Name of the room to explore.
            furniture_limit: Maximum number of furniture to visit (None = all).
            randomize_furniture: Whether to randomize the order of furniture visits.
            make_video: Whether to save video (ignored if live_display is enabled).
            on_furniture_visit: Callback after visiting each furniture.

        Returns:
            RoomExplorationResult with details of the exploration.
        """
        result = RoomExplorationResult(room_name=room_name)

        # Get furniture in this room
        furniture_list = self.get_furniture_in_room(room_name)

        if not furniture_list:
            cprint(f"  No furniture found in {room_name}", "yellow")
            result.error_message = f"No furniture in {room_name}"
            return result

        # Optionally randomize and limit furniture
        if randomize_furniture:
            furniture_list = random.sample(furniture_list, len(furniture_list))

        if furniture_limit is not None:
            furniture_list = furniture_list[:furniture_limit]

        cprint(f"  Exploring {len(furniture_list)} furniture items in {room_name}", "blue")

        # Visit each furniture
        for idx, furniture in enumerate(furniture_list):
            furniture_name = furniture.name
            cprint(f"    [{idx + 1}/{len(furniture_list)}] Navigating to {furniture_name}", "yellow")

            response_msg, frames, success = self.navigate_to_furniture(
                furniture_name,
                room_name=room_name,
                make_video=make_video,
                vid_postfix=f"{room_name}_{furniture_name}_",
            )

            result.furniture_visited.append(furniture_name)
            result.navigation_results.append((furniture_name, response_msg, success))
            result.frames.extend(frames)

            if not success:
                cprint(f"      Failed: {response_msg}", "red")
            else:
                cprint(f"      Success: {response_msg}", "green")

            # Call callback if provided
            if on_furniture_visit is not None:
                on_furniture_visit(room_name, furniture_name, response_msg)

        return result

    def explore_all_rooms(
        self,
        furniture_per_room: Optional[int] = None,
        randomize_furniture: bool = True,
        randomize_rooms: bool = False,
        room_filter: Optional[Callable[[str], bool]] = None,
        make_video: bool = False,
        on_room_enter: Optional[Callable[[str, int, int], None]] = None,
        on_room_exit: Optional[Callable[[str, RoomExplorationResult], None]] = None,
        on_furniture_visit: Optional[Callable[[str, str, str], None]] = None,
    ) -> FullExplorationResult:
        """
        Explore all rooms in the scene.

        Args:
            furniture_per_room: Max furniture to visit per room (None = all).
            randomize_furniture: Whether to randomize furniture order in each room.
            randomize_rooms: Whether to randomize the order of room visits.
            room_filter: Optional function to filter rooms. Return True to include.
            make_video: Whether to save videos (ignored if live_display is enabled).
            on_room_enter: Callback when entering a room.
            on_room_exit: Callback when exiting a room.
            on_furniture_visit: Callback after visiting furniture.

        Returns:
            FullExplorationResult with complete exploration data.
        """
        full_result = FullExplorationResult()

        # Get all rooms
        rooms = self.get_all_rooms()

        # Apply room filter if provided
        if room_filter is not None:
            rooms = [r for r in rooms if room_filter(r.name)]

        # Optionally randomize room order
        if randomize_rooms:
            rooms = random.sample(rooms, len(rooms))

        full_result.total_rooms = len(rooms)

        cprint(f"\n{'=' * 60}", "green")
        cprint(f"Starting exploration of {len(rooms)} rooms", "green")
        if self.live_display_enabled:
            cprint("  Live X11 display enabled - press 'q' to quit", "blue")
        cprint(f"{'=' * 60}\n", "green")

        try:
            # Explore each room
            for room_idx, room in enumerate(rooms):
                room_name = room.name

                cprint(f"\n=== Room {room_idx + 1}/{len(rooms)}: {room_name} ===", "blue")

                # Call room enter callback
                if on_room_enter is not None:
                    on_room_enter(room_name, room_idx, len(rooms))

                # Explore the room
                room_result = self.explore_room(
                    room_name=room_name,
                    furniture_limit=furniture_per_room,
                    randomize_furniture=randomize_furniture,
                    make_video=make_video,
                    on_furniture_visit=on_furniture_visit,
                )

                full_result.rooms_explored.append(room_result)
                full_result.total_furniture_visited += len(room_result.furniture_visited)
                full_result.all_frames.extend(room_result.frames)

                # Call room exit callback
                if on_room_exit is not None:
                    on_room_exit(room_name, room_result)

        finally:
            # Always close the display when done
            if self._live_display:
                self._live_display.close()

        cprint(f"\n{'=' * 60}", "green")
        cprint("Exploration complete!", "green")
        cprint(f"  Rooms explored: {len(full_result.rooms_explored)}", "blue")
        cprint(f"  Total furniture visited: {full_result.total_furniture_visited}", "blue")
        cprint(f"  Total frames collected: {len(full_result.all_frames)}", "blue")
        cprint(f"{'=' * 60}\n", "green")

        return full_result

    def explore_rooms_by_name(
        self,
        room_names: List[str],
        furniture_per_room: Optional[int] = None,
        randomize_furniture: bool = True,
        make_video: bool = False,
        on_furniture_visit: Optional[Callable[[str, str, str], None]] = None,
    ) -> FullExplorationResult:
        """Explore specific rooms by name."""
        return self.explore_all_rooms(
            furniture_per_room=furniture_per_room,
            randomize_furniture=randomize_furniture,
            room_filter=lambda name: name in room_names,
            make_video=make_video,
            on_furniture_visit=on_furniture_visit,
        )
