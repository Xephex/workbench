import pathlib, shutil, subprocess, sys, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


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


    def test_sparse_clone_fetches_its_own_package(self):
        """`git clone --sparse` checks out top-level files only; the launcher must add workbench/."""
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src = d / "src"
            src.mkdir()
            for name in ("workbench.py", "catalog.toml"):
                shutil.copy(ROOT / name, src / name)
            shutil.copytree(ROOT / "workbench", src / "workbench", ignore=shutil.ignore_patterns("__pycache__"))
            git("init", "-q", "-b", "main", cwd=src)
            git("add", ".", cwd=src)
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x", cwd=src)
            git("config", "uploadpack.allowFilter", "true", cwd=src)
            clone = d / "clone"
            git("clone", "-q", "--filter=blob:none", "--sparse", src.as_uri(), str(clone), cwd=d)
            self.assertFalse((clone / "workbench").exists())
            r = subprocess.run([sys.executable, str(clone / "workbench.py"), "status"],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Tier:", r.stdout)
            self.assertTrue((clone / "workbench" / "cli.py").exists())


if __name__ == "__main__":
    unittest.main()
