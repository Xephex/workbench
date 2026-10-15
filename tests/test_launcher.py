import pathlib, subprocess, sys, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def test_closed_pipe_exits_quietly(self):
        """`workbench.py status | head -1` must not print a traceback."""
        p = subprocess.Popen([sys.executable, str(ROOT / "workbench.py"), "status"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.stdout.readline()
        p.stdout.close()
        err = p.stderr.read().decode()
        p.wait(timeout=60)
        self.assertNotIn("Traceback", err)
        self.assertNotIn("BrokenPipeError", err)


if __name__ == "__main__":
    unittest.main()
