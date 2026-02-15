#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""Simple communication tool that lets agents exchange short text messages."""

from typing import List, Tuple

from habitat_llm.tools import PerceptionTool
from habitat_llm.utils.grammar import FREE_TEXT


class CommunicationTool(PerceptionTool):
    """
    Allows an agent to send messages to, or read messages from, its teammates.
    The actual data transport is handled by the EnvironmentInterface, which
    stores per-agent queues that persist across planner steps.
    """

    def __init__(self, skill_config):
        super().__init__(skill_config.name)
        self.skill_config = skill_config
        self._read_synonyms = {"", "read", "listen", "check", "receive"}
        self.allowed_targets = None  # None = unrestricted, List[int] = allowed recipient UIDs

    def set_environment(self, env_interface):
        self.env_interface = env_interface

    @property
    def description(self) -> str:
        return self.skill_config.description

    def process_high_level_action(
        self, input_query: str, observations: dict
    ) -> Tuple[None, str]:
        super().process_high_level_action(input_query, observations)
        if not self.env_interface:
            raise ValueError("Environment Interface not set for CommunicationTool")

        normalized = (input_query or "").strip()
        lowered = normalized.lower()

        if lowered in self._read_synonyms:
            pending = self.env_interface.consume_agent_messages(self.agent_uid)
            if len(pending) == 0:
                return (
                    None,
                    "No new teammate messages. Updates now flow into your context automatically whenever your partner speaks.",
                )
            formatted = "\n".join(
                [f"Agent_{msg['from']} said: {msg['message']}" for msg in pending]
            )
            return (
                None,
                "Messages are already appended to your context automatically, but here is the latest queue:\n"
                + formatted,
            )

        if normalized == "":
            return None, "Provide a message and recipients."

        # Parse targeted communication args
        valid_uids = list(self.env_interface.agent_uids) if self.env_interface.agent_uids else list(self.env_interface.world_graph.keys())
        message, target_uids = self.env_interface.parse_communicate_args(normalized, valid_uids)

        if not message:
            return None, "Provide a message to send."

        # Enforce message_targets restrictions
        if self.allowed_targets is not None:
            if target_uids is None:
                # Broadcast — narrow to only allowed targets
                target_uids = [uid for uid in self.allowed_targets if uid != self.agent_uid]
            else:
                # DM — filter to only allowed targets
                target_uids = [uid for uid in target_uids if uid in self.allowed_targets]

            if not target_uids:
                allowed_names = ", ".join(f"Agent_{uid}" for uid in self.allowed_targets)
                return None, f"You can only message: {allowed_names}. None of your specified recipients are allowed."

        self.env_interface.post_agent_message(self.agent_uid, message, target_uids=target_uids)

        if target_uids is None:
            recipient_desc = "all agents"
        else:
            recipient_desc = ", ".join(f"Agent_{uid}" for uid in target_uids)

        return (
            None,
            f'Message delivered to {recipient_desc}. They will see "Agent_{self.agent_uid} said: {message}" in their context automatically.',
        )

    @property
    def argument_types(self) -> List[str]:
        return [FREE_TEXT]
