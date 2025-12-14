"""
Habitat-integrated task runner for EMTOM benchmark.

Runs EMTOM tasks within the Habitat simulator with video recording support.
Uses the existing habitat_llm video recording utilities (DebugVideoUtil, FirstPersonVideoRecorder).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from emtom.benchmark.task_runner import (
    TaskRunner,
    TaskResult,
    TaskStatus,
    StepResult,
    RunConfig,
)

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface


@dataclass
class HabitatRunConfig(RunConfig):
    """Configuration for Habitat-integrated task execution."""
    save_video: bool = True
    video_fps: int = 30
    output_dir: str = "data/emtom/benchmark"
    play_video: bool = False
    save_fpv: bool = True  # Save first-person view videos


class HabitatTaskRunner(TaskRunner):
    """
    Task runner that integrates with Habitat simulator.

    Extends TaskRunner with:
    - Habitat EnvironmentInterface integration
    - Video recording using DebugVideoUtil and FirstPersonVideoRecorder
    - Image observation access for agent decision making
    """

    def __init__(
        self,
        env_interface: "EnvironmentInterface",
        config: Optional[HabitatRunConfig] = None,
    ):
        """
        Initialize the Habitat task runner.

        Args:
            env_interface: Habitat EnvironmentInterface instance
            config: Configuration for task execution
        """
        self.env_interface = env_interface
        self.habitat_config = config or HabitatRunConfig()
        super().__init__(self.habitat_config)

        # Video recording utilities from habitat_llm
        self._dvu = None  # DebugVideoUtil
        self._fpv_recorder = None  # FirstPersonVideoRecorder
        self._setup_video_recording()

        # Store observations for potential image access
        self._last_observations: Dict[str, Any] = {}

    def _setup_video_recording(self) -> None:
        """Initialize video recording utilities."""
        if not self.habitat_config.save_video:
            return

        try:
            from habitat_llm.examples.example_utils import (
                DebugVideoUtil,
                FirstPersonVideoRecorder,
            )

            # Ensure output directory exists
            os.makedirs(self.habitat_config.output_dir, exist_ok=True)

            # Initialize DebugVideoUtil for third-person split-screen video
            self._dvu = DebugVideoUtil(
                self.env_interface,
                self.habitat_config.output_dir,
                unique_postfix=True,
            )

            # Initialize FirstPersonVideoRecorder for per-agent FPV videos
            if self.habitat_config.save_fpv:
                try:
                    self._fpv_recorder = FirstPersonVideoRecorder(
                        self.env_interface,
                        output_dir=self.habitat_config.output_dir,
                        fps=self.habitat_config.video_fps,
                    )
                except Exception as e:
                    print(f"[EMTOM] Failed to initialize FPV recorder: {e}")
                    self._fpv_recorder = None

        except ImportError as e:
            print(f"[EMTOM] Warning: Could not import video utilities: {e}")
            self._dvu = None
            self._fpv_recorder = None

    def get_observations(self) -> Dict[str, Any]:
        """Get current observations from the Habitat environment."""
        self._last_observations = self.env_interface.get_observations()
        return self._last_observations

    def get_image_observation(
        self,
        agent_id: str = "agent_0",
        camera_type: str = "third_rgb",
    ) -> Optional[np.ndarray]:
        """
        Get RGB image observation for an agent.

        Args:
            agent_id: Agent ID to get observation for
            camera_type: Type of camera ("third_rgb", "head_rgb", "jaw_rgb")

        Returns:
            RGB image as numpy array (H, W, 3), or None if not available
        """
        obs = self.get_observations()

        # Build possible observation keys
        possible_keys = [
            f"{agent_id}_{camera_type}",
            f"{agent_id}_articulated_agent_{camera_type}",
            camera_type,
        ]

        for key in possible_keys:
            if key in obs:
                img = obs[key]
                return self._to_numpy_image(img)

        return None

    def _to_numpy_image(self, img: Any) -> np.ndarray:
        """Convert observation to numpy image array."""
        if hasattr(img, 'cpu'):
            img = img.cpu()
        if hasattr(img, 'numpy'):
            img = img.numpy()
        img = np.array(img)

        # Handle batch dimension
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]

        # Handle channel-first format
        if img.ndim == 3 and img.shape[0] in (3, 4):
            img = np.transpose(img, (1, 2, 0))

        # Convert to uint8
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Keep only RGB channels
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        return np.ascontiguousarray(img)

    def _record_frame(self, observations: Dict[str, Any], actions: Dict[int, Any] = None) -> None:
        """Record a video frame from current observations."""
        if actions is None:
            actions = {}

        # Record third-person split-screen frame
        if self._dvu is not None:
            try:
                self._dvu._store_for_video(observations, actions, popup_images={})
            except Exception as e:
                print(f"[EMTOM] Failed to store third-person frame: {e}")

        # Record first-person frames
        if self._fpv_recorder is not None:
            try:
                self._fpv_recorder.record_step(observations)
            except Exception as e:
                # Only log once per type of error
                if not hasattr(self, '_fpv_error_logged'):
                    print(f"[EMTOM] Failed to record FPV frame: {e}")
                    self._fpv_error_logged = True

    def _save_videos(self, task_id: str) -> Dict[str, str]:
        """
        Save recorded videos.

        Returns:
            Dict mapping video type to file path
        """
        video_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        postfix = f"{task_id}_{timestamp}"

        # Save third-person video
        if self._dvu is not None and self._dvu.frames:
            try:
                self._dvu._make_video(play=self.habitat_config.play_video, postfix=postfix)
                video_dir = os.path.join(self.habitat_config.output_dir, "videos")
                video_paths["third_person"] = os.path.join(video_dir, f"video-{postfix}.mp4")
            except Exception as e:
                print(f"[EMTOM] Failed to save third-person video: {e}")

        # Save first-person videos
        if self._fpv_recorder is not None:
            try:
                fpv_paths = self._fpv_recorder.save(postfix=postfix)
                for agent_name, path in fpv_paths.items():
                    video_paths[f"fpv_{agent_name}"] = path
                    print(f"[EMTOM] Saved FPV video for {agent_name}: {path}")
            except Exception as e:
                print(f"[EMTOM] Failed to save FPV videos: {e}")

        return video_paths

    def _clear_video_buffers(self) -> None:
        """Clear video recording buffers for a new task."""
        if self._dvu is not None:
            self._dvu.frames.clear()
        if self._fpv_recorder is not None:
            self._fpv_recorder._frames = {}

    def run(self) -> TaskResult:
        """
        Execute the task with video recording.

        Returns:
            TaskResult with success/failure status and metrics
        """
        if not self.current_task or not self.world:
            raise RuntimeError("Must call setup_task before run")

        start_time = time.time()
        task = self.current_task
        steps_taken: List[StepResult] = []

        # Clear video buffers for new task
        self._clear_video_buffers()

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"RUNNING TASK: {task.title}")
            print(f"{'='*60}")
            print(f"Agents: {list(self.agents.keys())}")
            print(f"Mechanics: {task.required_mechanics}")
            print(f"Goal: {task.success_condition.description}")
            print()

        # Record initial frame
        obs = self.get_observations()
        self._record_frame(obs, {})

        # Main execution loop
        while self._step_count < self.config.max_steps:
            # Check time limit
            if task.success_condition.time_limit:
                if self._step_count >= task.success_condition.time_limit:
                    result = self._create_result(
                        TaskStatus.TIMEOUT,
                        steps_taken,
                        time.time() - start_time,
                        f"Exceeded time limit of {task.success_condition.time_limit} steps",
                    )
                    self._save_videos(task.task_id)
                    return result

            # Execute one step
            step_result = self._execute_step()
            steps_taken.append(step_result)

            # Record frame after step with action info
            obs = self.get_observations()
            # Build action dict for video overlay
            actions = {}
            for i, (agent_id, action) in enumerate(step_result.agent_actions.items()):
                actions[i] = (action.action_name, action.target or "")
            self._record_frame(obs, actions)

            # Check success condition
            if self._check_success():
                if self.config.verbose:
                    print(f"\n[SUCCESS] Task completed in {self._step_count} steps!")
                result = self._create_result(
                    TaskStatus.SUCCESS,
                    steps_taken,
                    time.time() - start_time,
                )
                video_paths = self._save_videos(task.task_id)
                if video_paths:
                    result.metadata["video_paths"] = video_paths
                return result

            # Check failure conditions
            failure_reason = self._check_failure()
            if failure_reason:
                if self.config.verbose:
                    print(f"\n[FAILURE] {failure_reason}")
                result = self._create_result(
                    TaskStatus.FAILURE,
                    steps_taken,
                    time.time() - start_time,
                    failure_reason,
                )
                video_paths = self._save_videos(task.task_id)
                if video_paths:
                    result.metadata["video_paths"] = video_paths
                return result

            self._step_count += 1

            if self.config.step_delay > 0:
                time.sleep(self.config.step_delay)

        # Max steps exceeded
        result = self._create_result(
            TaskStatus.TIMEOUT,
            steps_taken,
            time.time() - start_time,
            f"Exceeded maximum steps ({self.config.max_steps})",
        )
        video_paths = self._save_videos(task.task_id)
        if video_paths:
            result.metadata["video_paths"] = video_paths
        return result


def create_habitat_runner(
    config_path: str = "habitat_llm/conf",
    config_name: str = "examples/planner_multi_agent_demo_config",
    overrides: List[str] = None,
    run_config: HabitatRunConfig = None,
) -> HabitatTaskRunner:
    """
    Create a HabitatTaskRunner with proper Hydra configuration.

    Args:
        config_path: Path to Hydra config directory
        config_name: Name of the config file
        overrides: List of Hydra overrides
        run_config: Configuration for task execution

    Returns:
        Configured HabitatTaskRunner instance
    """
    from hydra import compose, initialize_config_dir

    from habitat_llm.agent.env import (
        EnvironmentInterface,
        register_actions,
        register_measures,
        register_sensors,
    )
    from habitat_llm.utils import setup_config, fix_config

    # Initialize Hydra
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(config_path)):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    fix_config(cfg)
    cfg = setup_config(cfg, seed=47668090)

    # Enable video saving
    cfg.evaluation.save_video = True

    # Register Habitat components
    register_sensors(cfg)
    register_actions(cfg)
    register_measures(cfg)

    # Create environment interface
    env_interface = EnvironmentInterface(cfg)

    # Create runner
    run_config = run_config or HabitatRunConfig(
        save_video=True,
        output_dir=cfg.paths.results_dir,
    )

    return HabitatTaskRunner(env_interface, run_config)
