#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio
import numpy as np

from habitat_llm.agent.env import EnvironmentInterface


class DebugVideoUtil:
    """
    This class provides an interface wrapper for creating, saving, and viewing third person videos of individual skill runs using the EnvironmentInterface API.

    For example, see `execute_skill` function below.
    NOTE: This code was largely adapted from the evaluation_runner.py
    """

    def __init__(
        self, env_interface_arg: EnvironmentInterface, output_dir: str, unique_postfix: bool = False
    ) -> None:
        """
        Construct the DebugVideoUtil instance from an EnvironmentInterface.

        :param env_interface_arg: The EnvironmentInterface instance.
        :param output_dir: The desired directory for saving output frames and videos.
        """

        self.env_interface = env_interface_arg

        # Declare container to store frames used for generating video
        self.frames: List[Any] = []

        self.output_dir = output_dir
        self.unique_postfix = unique_postfix

        self.num_agents = 0
        for _agent_conf in self.env_interface.conf.evaluation.agents.values():
            self.num_agents += 1

    def __get_combined_frames(self, batch: Dict[str, Any]) -> np.ndarray:
        """
        For each agent, extract the observation from the "third_rgb" sensor and merge them into a single split-screen image.

        :param batch: A dict mapping observation names to values.
        :return: The composite image as a numpy array.
        """
        # Extract agent frames from third_rgb observations
        images = []
        third_rgb_keys = [k for k in batch.keys() if "third_rgb" in k]

        for obs_name in sorted(third_rgb_keys):  # Sort for consistent ordering
            obs_value = batch[obs_name]
            if self.num_agents == 1:
                if "0" in obs_name or "main_agent" in obs_name:
                    images.append(obs_value)
            else:
                images.append(obs_value)

        # Handle case where no images found
        if not images:
            # Try to find ANY rgb observation as fallback
            rgb_keys = [k for k in batch.keys() if "rgb" in k.lower()]
            for obs_name in sorted(rgb_keys):
                obs_value = batch[obs_name]
                if hasattr(obs_value, 'shape'):
                    images.append(obs_value)
                    if len(images) >= self.num_agents:
                        break

            if not images:
                raise ValueError(
                    f"No third_rgb observations found in batch. "
                    f"Available keys: {list(batch.keys())}, num_agents={self.num_agents}"
                )

        # Extract dimensions of the first image
        # Handle different tensor formats (batch, channels, height, width) vs (height, width, channels)
        first_img = images[0]
        if hasattr(first_img, 'cpu'):
            first_img = first_img.cpu()
        if hasattr(first_img, 'numpy'):
            first_img = first_img.numpy()

        # Determine shape based on format
        if len(first_img.shape) == 4:
            # Batched: (batch, height, width, channels) or (batch, channels, height, width)
            height, width = first_img.shape[1:3]
        elif len(first_img.shape) == 3:
            # Single: (height, width, channels) or (channels, height, width)
            if first_img.shape[0] in (3, 4):  # channels first
                height, width = first_img.shape[1:3]
            else:  # channels last
                height, width = first_img.shape[0:2]
        else:
            raise ValueError(f"Unexpected image shape: {first_img.shape}")

        # Create an empty canvas to hold the concatenated images
        concat_image = np.zeros((height, width * len(images), 3), dtype=np.uint8)

        # Iterate through the images and concatenate them horizontally
        for i, image in enumerate(images):
            if hasattr(image, 'cpu'):
                image = image.cpu()
            if hasattr(image, 'numpy'):
                image = image.numpy()

            # Handle different formats
            if len(image.shape) == 4:
                image = image[0]  # Remove batch dimension
            if image.shape[0] in (3, 4) and len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

            # Ensure uint8
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # Take only RGB channels
            if image.shape[-1] > 3:
                image = image[..., :3]

            concat_image[:, i * width : (i + 1) * width] = image

        return concat_image

    def _store_for_video(
        self,
        observations: Dict[str, Any],
        hl_actions: Dict[int, Any],
        popup_images: Dict[int, str] = None,
    ) -> None:
        """
        Store a video with observations and text from an observation dict and an agent to action metadata dict.
        NOTE: Could probably go into utils?

        :param observations: A dict mapping observation names to values.
        :param hl_actions: A dict mapping agent action indices to actions.
        """
        frames_concat = self.__get_combined_frames(observations)
        frames_concat = np.ascontiguousarray(frames_concat)

        for idx, action in hl_actions.items():
            # text = f"Agent_{id}:{action[0]}[{action[1]}]"
            agent_name = "Human" if str(idx) == "1" else "Robot"
            text = f"{agent_name}: {action[0]}[{action[1]}]"
            frames_concat = cv2.putText(
                frames_concat,
                text,
                (20, (int(idx) + 1) * 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

        # Overlay popups if provided (per agent, left/right). Align with eval runner style.
        if popup_images:
            # we assume two agents max for overlay placement
            for agent_idx, path in popup_images.items():
                try:
                    popup = cv2.imread(path)
                    if popup is None:
                        continue
                    # resize popup to fit in the corner
                    ph, pw = popup.shape[:2]
                    scale = 0.3
                    popup = cv2.resize(popup, (int(pw * scale), int(ph * scale)))
                    ph, pw = popup.shape[:2]
                    # Position popups along the bottom to avoid covering the main view.
                    if int(agent_idx) == 0:
                        y1 = frames_concat.shape[0] - 10
                        y0 = y1 - ph
                        x0, x1 = 10, 10 + pw
                    else:
                        y1 = frames_concat.shape[0] - 10
                        y0 = y1 - ph
                        x1 = frames_concat.shape[1] - 10
                        x0 = x1 - pw
                    frames_concat[y0:y1, x0:x1] = popup
                except Exception:
                    continue

        self.frames.append(frames_concat)
        return

    def _make_video(self, play: bool = True, postfix: str = "", fps: int = 30) -> None:
        """
        Makes a video from a pre-processed set of frames using imageio and saves it to the output directory.

        :param play: Whether or not to play the video immediately.
        :param postfix: An optional postfix for the video file name.
        :param fps: Frames per second for the video (default 30).
        """
        if not self.frames:
            print("No frames to write; skipping video.")
            return
        frames = list(self.frames)
        print(f"[DebugVideoUtil] Total frames to write: {len(frames)}")
        if len(frames) == 1:
            frames = frames * fps  # pad to ~1s so the video isn't empty
        extra = f"-{postfix}" if postfix else ""
        if self.unique_postfix:
            extra = f"{extra}-{int(time.time()*1000)}"
        out_file = f"{self.output_dir}/videos/video{extra}.mp4"
        print(f"Saving video to {out_file} ({len(frames)} frames @ {fps} FPS)")
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
        writer = imageio.get_writer(out_file, fps=fps, quality=4)
        for frame in frames:
            writer.append_data(frame)

        writer.close()
        duration = len(frames) / fps
        print(f"[DebugVideoUtil] Video saved: {duration:.1f}s duration")
        if play:
            print("     ...playing video, press 'q' to continue...")
            self.play_video(out_file)

    def play_video(self, filename: str) -> None:
        """
        Play and loop video from a filepath with cv2.

        :param filename: The filepath of the video.
        """
        cap = cv2.VideoCapture(filename)
        last_time = time.time()
        while cap.isOpened():
            if time.time() - last_time > 1.0 / 30:
                last_time = time.time()
                ret, frame = cap.read()
                # cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
                # cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

                if ret:
                    cv2.imshow("Image", frame)
                else:
                    # looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


class FirstPersonVideoRecorder:
    """
    Minimal helper to write first-person videos that mirror the trajectory logger
    camera selection. This avoids relying on the split-screen DebugVideoUtil.
    """

    def __init__(
        self,
        env_interface: EnvironmentInterface,
        output_dir: Optional[str] = None,
        fps: int = 30,
    ) -> None:
        self.env_interface = env_interface
        # Decide where to write outputs
        results_root = getattr(getattr(env_interface.conf, "paths", None), "results_dir", None)
        self.output_dir = output_dir or results_root or "outputs"
        self.fps = fps
        self._frames: Dict[str, List[np.ndarray]] = {}
        self._camera_keys: Dict[str, str] = self._build_camera_keys()

    def _build_camera_keys(self) -> Dict[str, str]:
        """
        Build a map from agent name to observation key using the trajectory config.
        This keeps the FPV video aligned with the saved trajectories.
        """
        traj_conf = getattr(self.env_interface.conf, "trajectory", None)
        if traj_conf is None:
            raise ValueError("Trajectory config not found; cannot determine FPV cameras.")
        agent_names = list(getattr(traj_conf, "agent_names", []))
        cam_prefixes = list(getattr(traj_conf, "camera_prefixes", []))
        if not agent_names or not cam_prefixes or len(agent_names) != len(cam_prefixes):
            raise ValueError("trajectory.agent_names and trajectory.camera_prefixes must be provided and have the same length.")

        camera_keys: Dict[str, str] = {}
        for agent_name, cam_prefix in zip(agent_names, cam_prefixes):
            if getattr(self.env_interface, "_single_agent_mode", False):
                key = f"{cam_prefix}_rgb"
            else:
                key = f"{agent_name}_{cam_prefix}_rgb"
            camera_keys[agent_name] = key
        return camera_keys

    @staticmethod
    def _to_uint8(frame: Any) -> np.ndarray:
        """
        Convert a tensor/array frame to channel-last uint8.
        """
        if hasattr(frame, "detach"):
            frame = frame.detach()
        if hasattr(frame, "cpu"):
            frame = frame.cpu()
        frame_np = np.array(frame)
        if frame_np.ndim == 4 and frame_np.shape[0] == 1:
            frame_np = frame_np[0]
        if frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
            frame_np = np.transpose(frame_np, (1, 2, 0))
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255)
            if frame_np.max() <= 1.0:
                frame_np = frame_np * 255.0
            frame_np = frame_np.astype(np.uint8)
        if frame_np.ndim != 3 or frame_np.shape[2] < 3:
            raise ValueError(f"Frame has invalid shape for video: {frame_np.shape}")
        return np.ascontiguousarray(frame_np[:, :, :3])

    def record_step(self, observations: Dict[str, Any]) -> None:
        """
        Capture first-person frames for all configured agents from a single step.
        """
        for agent_name, obs_key in self._camera_keys.items():
            if obs_key not in observations:
                raise KeyError(
                    f"Observation key '{obs_key}' missing for agent '{agent_name}'. "
                    f"Available keys: {list(observations.keys())}"
                )
            frame = self._to_uint8(observations[obs_key])
            self._frames.setdefault(agent_name, []).append(frame)

    def save(self, postfix: str = "") -> Dict[str, str]:
        """
        Write per-agent FPV videos to disk. Returns a map of agent name to path.
        """
        if not self._frames:
            raise ValueError("No frames recorded; call record_step before save().")
        video_dir = os.path.join(self.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        paths: Dict[str, str] = {}
        for agent_name, frames in self._frames.items():
            if not frames:
                continue
            filename = f"fpv_{agent_name}"
            if postfix:
                filename += f"_{postfix}"
            video_path = os.path.join(video_dir, f"{filename}.mp4")
            imageio.mimwrite(video_path, frames, fps=self.fps)
            paths[agent_name] = video_path
        # Clear stored frames after writing
        self._frames = {}
        return paths


def execute_skill(
    high_level_skill_actions: Dict[Any, Any],
    llm_env,
    make_video: bool = True,
    vid_postfix: str = "",
    play_video: bool = True,
) -> Tuple[Dict[Any, Any], Dict[Any, Any], List[Any]]:
    """
    Execute a high-level skill from a string (e.g. as produced by the planner).
    Can create and display a video of the running skill.

    :param high_level_skill_actions: The map of agent indices to actions. TODO: typing
    :param llm_env: The planner instance. TODO: typing
    :param make_video: whether or not to create, save, and display a video of the skill.
    :param vid_postfix: An optional postfix for the video file. For example, the action name.
    :param play_video: Whether or not to immediately play the generated video.
    :return: A tuple with two dict(the first contains responses per-agent skill, the second contains the number of skill steps taken) and a list of frames.
    """
    dvu = DebugVideoUtil(
        llm_env.env_interface, llm_env.env_interface.conf.paths.results_dir
    )

    # Get the env observations
    observations = llm_env.env_interface.get_observations()
    agent_idx = list(high_level_skill_actions.keys())[0]
    skill_name = high_level_skill_actions[agent_idx][0]

    # Set up the variables
    skill_steps = 0
    max_skill_steps = 1500
    skill_done = None

    # While loop for executing skills
    while not skill_done:
        # Check if the maximum number of steps is reached
        assert (
            skill_steps < max_skill_steps
        ), f"Maximum number of steps reached: {skill_name} skill fails."

        # Get low level actions and responses
        low_level_actions, responses = llm_env.process_high_level_actions(
            high_level_skill_actions, observations
        )

        assert (
            len(low_level_actions) > 0
        ), f"No low level actions returned. Response: {responses.values()}"

        # Check if the agent finishes
        if any(responses.values()):
            skill_done = True

        # Get the observations
        obs, reward, done, info = llm_env.env_interface.step(low_level_actions)
        observations = llm_env.env_interface.parse_observations(obs)

        if make_video:
            dvu._store_for_video(observations, high_level_skill_actions)

        # Increase steps
        skill_steps += 1

    if make_video and skill_steps > 1:
        dvu._make_video(postfix=vid_postfix, play=play_video)

    return responses, {"skill_steps": skill_steps}, dvu.frames
