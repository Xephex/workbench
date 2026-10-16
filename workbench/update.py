"""Pull the checkout this program is running from. Fast-forward only."""

from __future__ import annotations

import pathlib, subprocess


def git_update(root: pathlib.Path) -> tuple[int, str]:
    """Return a process code and the text to show.

    Local changes stop the pull. A folder that is not a checkout cannot update.
    """
    marker = root / ".git"
    if not marker.exists():
        return 1, "This folder is not a git checkout, so there is nothing to update."
    git = ["git", "-C", str(root)]
    status = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True)
    if status.returncode != 0:
        return status.returncode, (status.stderr or status.stdout or "git status failed").strip()
    if status.stdout.strip():
        return 1, "Update stopped because this checkout has local changes:\n" + status.stdout.rstrip()
    pull = subprocess.run([*git, "pull", "--ff-only"], capture_output=True, text=True)
    text = "\n".join(part.strip() for part in (pull.stdout, pull.stderr) if part and part.strip())
    return pull.returncode, text or "git pull finished."
