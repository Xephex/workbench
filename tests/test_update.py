import pathlib, subprocess, tempfile, unittest

from workbench.update import git_update


def git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


class UpdateTest(unittest.TestCase):
    def test_fast_forward_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            origin = tmp / "origin"
            origin.mkdir()
            git("init", "-q", "-b", "main", cwd=origin)
            (origin / "note.txt").write_text("one\n")
            git("add", "note.txt", cwd=origin)
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one", cwd=origin)
            clone = tmp / "clone"
            git("clone", "-q", origin.as_uri(), str(clone), cwd=tmp)
            (origin / "note.txt").write_text("two\n")
            git("add", "note.txt", cwd=origin)
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "two", cwd=origin)
            code, text = git_update(clone)
            self.assertEqual(code, 0, text)
            self.assertIn("two", (clone / "note.txt").read_text())
            self.assertNotIn("local changes", text)

    def test_local_changes_stop_the_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            origin = tmp / "origin"
            origin.mkdir()
            git("init", "-q", "-b", "main", cwd=origin)
            (origin / "note.txt").write_text("one\n")
            git("add", "note.txt", cwd=origin)
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "one", cwd=origin)
            clone = tmp / "clone"
            git("clone", "-q", origin.as_uri(), str(clone), cwd=tmp)
            (clone / "note.txt").write_text("local\n")
            (origin / "note.txt").write_text("two\n")
            git("add", "note.txt", cwd=origin)
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "two", cwd=origin)
            code, text = git_update(clone)
            self.assertEqual(code, 1)
            self.assertIn("local changes", text)
            self.assertEqual((clone / "note.txt").read_text(), "local\n")

    def test_not_a_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, text = git_update(pathlib.Path(tmp))
        self.assertEqual(code, 1)
        self.assertIn("not a git checkout", text)


if __name__ == "__main__":
    unittest.main()
