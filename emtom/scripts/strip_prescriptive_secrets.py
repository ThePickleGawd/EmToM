#!/usr/bin/env python3
"""Strip prescriptive communication instructions from cooperative task secrets.

Removes secret entries that tell agents exactly what to communicate, who to
message, or prescribe relay chains. Keeps role info, constraints, and
goal-critical information.

This forces agents to use genuine Theory of Mind to decide what to communicate.
"""

import json
import glob
import os
import re
import sys
import shutil


# Patterns that indicate prescriptive communication instructions
PRESCRIPTIVE_PATTERNS = [
    r"[Cc]oordination note:",
    r"[Cc]oordination fallback:",
    r"[Cc]oordination hint:",
    r"message agent_\d",
    r"tell agent_\d",
    r"send agent_\d",
    r"relay.*to agent_\d",
    r"forward.*to agent_\d",
    r"inform agent_\d",
    r"notify agent_\d",
    r"ask agent_\d.*(message|send|tell|relay|forward)",
    r"agent_\d will.*relay",
    r"agent_\d will.*message",
    r"agent_\d will.*forward",
    r"agent_\d will.*send",
    r"'[^']+is (OPEN|CLOSED|TRUE|on|off|inside|on top)'",  # Exact message to send
    r'"[^"]+is (OPEN|CLOSED|TRUE|on|off|inside|on top)"',
    r"like '[^']+'",  # "like 'cabinet_26 is OPEN'"
]


def is_prescriptive(secret_line: str) -> bool:
    """Check if a secret line is prescriptive communication instruction."""
    for pattern in PRESCRIPTIVE_PATTERNS:
        if re.search(pattern, secret_line, re.IGNORECASE):
            return True
    return False


def strip_secrets(task_data: dict) -> dict:
    """Remove prescriptive communication lines from agent secrets."""
    modified = json.loads(json.dumps(task_data))  # deep copy

    for agent_id, secrets in modified.get("agent_secrets", {}).items():
        if isinstance(secrets, list):
            filtered = [s for s in secrets if not is_prescriptive(s)]
            # Keep at least one secret per agent
            if filtered:
                modified["agent_secrets"][agent_id] = filtered
            # If all secrets were prescriptive, keep the first one (role info)
            elif secrets:
                modified["agent_secrets"][agent_id] = [secrets[0]]
        elif isinstance(secrets, str):
            lines = secrets.split("\n")
            filtered = [l for l in lines if not is_prescriptive(l)]
            modified["agent_secrets"][agent_id] = "\n".join(filtered) if filtered else secrets

    return modified


def main():
    tasks_dir = sys.argv[1] if len(sys.argv) > 1 else "data/emtom/tasks"
    dry_run = "--dry-run" in sys.argv
    backup_dir = os.path.join(tasks_dir, "backup_pre_strip")

    print(f"Stripping prescriptive secrets from cooperative tasks in {tasks_dir}")
    print(f"Dry run: {dry_run}")

    modified = 0
    unchanged = 0

    for f in sorted(glob.glob(os.path.join(tasks_dir, "*.json"))):
        basename = os.path.basename(f)
        if basename.startswith("202602"):  # skip salvaged
            continue

        with open(f) as fh:
            data = json.load(fh)

        if data.get("category") != "cooperative":
            continue

        # Check calibration — only modify tasks that GPT-5.2 passes
        cal = data.get("calibration", {})
        passes_gpt52 = False
        if isinstance(cal, list):
            for entry in cal:
                if not isinstance(entry, dict):
                    continue
                models = entry.get("agent_models", {})
                if not any("gpt-5.2" in str(v) for v in models.values()):
                    continue
                res = entry.get("results", {})
                if res.get("passed") or res.get("progress", 0) >= 1.0:
                    passes_gpt52 = True
                mg = res.get("main_goal", {})
                if isinstance(mg, dict) and mg.get("passed"):
                    passes_gpt52 = True
                break

        # Also check old format
        if isinstance(cal, dict) and "gpt-5.2" in cal:
            entry = cal["gpt-5.2"]
            if isinstance(entry, dict):
                if entry.get("passed") or entry.get("percent_complete", 0) >= 1.0:
                    passes_gpt52 = True

        if not passes_gpt52:
            # Skip tasks that already fail
            continue

        # Strip prescriptive secrets
        stripped = strip_secrets(data)

        # Check if anything changed
        if json.dumps(stripped["agent_secrets"]) == json.dumps(data["agent_secrets"]):
            unchanged += 1
            continue

        # Count removed lines
        orig_count = sum(len(v) if isinstance(v, list) else v.count("\n") + 1
                        for v in data.get("agent_secrets", {}).values())
        new_count = sum(len(v) if isinstance(v, list) else v.count("\n") + 1
                       for v in stripped.get("agent_secrets", {}).values())
        removed = orig_count - new_count

        if dry_run:
            print(f"  [DRY-RUN] {basename[:55]} — remove {removed} prescriptive lines")
        else:
            # Backup original
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            backup_path = os.path.join(backup_dir, basename)
            if not os.path.exists(backup_path):
                shutil.copy2(f, backup_path)

            # Write modified
            with open(f, "w") as fh:
                json.dump(stripped, fh, indent=2)
            print(f"  [STRIPPED] {basename[:55]} — removed {removed} prescriptive lines")

        modified += 1

    print(f"\nModified: {modified}, Unchanged: {unchanged}")
    if dry_run:
        print("[DRY-RUN — no files changed]")
    elif modified > 0:
        print(f"Backups in: {backup_dir}")


if __name__ == "__main__":
    main()
