"""
Task-generation command surface for external SWE agents.

Usage:
    python -m emtom.cli.taskgen --working-dir DIR status
    python -m emtom.cli.taskgen --working-dir DIR new_scene 3 [--keep]
    python -m emtom.cli.taskgen --working-dir DIR judge
    python -m emtom.cli.taskgen --working-dir DIR verify_golden_trajectory
    python -m emtom.cli.taskgen --working-dir DIR test_task
    python -m emtom.cli.taskgen --working-dir DIR submit_task
    python -m emtom.cli.taskgen --working-dir DIR finish
    python -m emtom.cli.taskgen --working-dir DIR fail "reason"
"""

from __future__ import annotations

import argparse

from emtom.cli import print_result
from emtom.task_gen.session import TaskGenSession


def main() -> None:
    parser = argparse.ArgumentParser(description="External-agent task-generation commands")
    parser.add_argument("--working-dir", required=True, help="Task generation working directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    new_scene_parser = subparsers.add_parser("new_scene", help="Load or reload a scene")
    new_scene_parser.add_argument("num_agents", type=int, help="Number of agents")
    new_scene_parser.add_argument("--keep", action="store_true", help="Keep current scene/task")

    subparsers.add_parser("judge", help="Judge the current task")
    subparsers.add_parser(
        "verify_golden_trajectory", help="Regenerate and verify the golden trajectory"
    )
    subparsers.add_parser("test_task", help="Run standard and baseline calibration runs")
    subparsers.add_parser("submit_task", help="Submit the current task")
    subparsers.add_parser("status", help="Show current task-generation state")
    subparsers.add_parser("finish", help="Mark the run complete")

    fail_parser = subparsers.add_parser("fail", help="Mark the run failed")
    fail_parser.add_argument("reason", help="Failure reason")

    args = parser.parse_args()
    session = TaskGenSession(args.working_dir)

    if args.command == "new_scene":
        result = session.new_scene(args.num_agents, keep=args.keep)
    elif args.command == "judge":
        result = session.judge()
    elif args.command == "verify_golden_trajectory":
        result = session.verify_golden_trajectory()
    elif args.command == "test_task":
        result = session.test_task()
    elif args.command == "submit_task":
        result = session.submit_task()
    elif args.command == "status":
        result = session.status()
    elif args.command == "finish":
        result = session.finish()
    else:
        result = session.fail(args.reason)

    print_result(result)


if __name__ == "__main__":
    main()
