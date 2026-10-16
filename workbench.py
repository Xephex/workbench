r"""Offline LLM workbench.

    python workbench.py setup      detect this machine, install, choose and fetch models
    python workbench.py models     add or remove models
    python workbench.py serve      local server on 127.0.0.1:8080
    python workbench.py chat       talk to a model through that server
    python workbench.py ask        one question through that server
    python workbench.py status     what this machine is and what is installed

On Windows use:  py -3.14 workbench.py setup
"""

import os, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent
NEED = (3, 14)


def find_python_314():
    """Return a command list for an installed Python 3.14, or None."""
    cands = []
    if shutil.which("py"):
        cands.append(["py", "-3.14"])
    for name in ("python3.14", "python3", "python"):
        p = shutil.which(name)
        if p:
            cands.append([p])
    for c in cands:
        try:
            out = subprocess.run([*c, "-c", "import sys;print(sys.version_info[:2]==(3,14))"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out == "True":
            return c
    return None


def ensure_python():
    if sys.version_info[:2] == NEED:
        return
    other = find_python_314()
    if other and not os.environ.get("WORKBENCH_REEXEC"):
        print(f"You started this with Python {sys.version.split()[0]}; switching to Python 3.14.", flush=True)
        env = dict(os.environ, WORKBENCH_REEXEC="1")
        sys.exit(subprocess.run([*other, str(ROOT / "workbench.py"), *sys.argv[1:]], env=env).returncode)
    print(f"The workbench needs Python 3.14. You are on {sys.version.split()[0]}, and no Python 3.14 was found.")
    print("Install Python 3.14 (python.org, or the Microsoft Store on Windows), then run:")
    print("  py -3.14 workbench.py setup      (Windows)")
    print("  python3.14 workbench.py setup    (Linux, macOS)")
    sys.exit(1)


def ensure_package():
    """A `git clone --sparse` checks out only top-level files. Fetch the package folder."""
    if (ROOT / "workbench" / "__init__.py").exists():
        return
    if not (ROOT / ".git").exists():
        print(f"The folder {ROOT / 'workbench'} is missing. Download the workbench again with:")
        print("  git clone --filter=blob:none --sparse https://github.com/Xephex/workbench")
        sys.exit(1)
    print("Fetching the workbench program files (first run only)...", flush=True)
    r = subprocess.run(["git", "-C", str(ROOT), "sparse-checkout", "add", "workbench", "tools"])
    if r.returncode != 0 or not (ROOT / "workbench" / "__init__.py").exists():
        print("git could not fetch the program files (message above). Check the network and run this again.")
        sys.exit(1)


if __name__ == "__main__":
    ensure_python()
    ensure_package()
    sys.path.insert(0, str(ROOT))
    from workbench import cli

    try:
        code = cli.main(sys.argv[1:])
        sys.stdout.flush()
    except BrokenPipeError:  # output piped into something that stopped reading (e.g. `| head`)
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 0
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue; downloads resume where they stopped.")
        code = 130
    sys.exit(code)
