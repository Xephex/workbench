import os, pathlib, tempfile, unittest
from unittest import mock

from tests import gitfixture
from workbench import catalog, fetch


class FetchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.data = os.urandom(10_000)
        (self.f,) = gitfixture.make_repo(self.base / "remote", "Xephex/models", {"m.gguf": self.data}, chunk=3_000)
        self.env = mock.patch.dict(os.environ, {"WORKBENCH_REPO_BASE": (self.base / "remote").as_uri()})
        self.env.start()
        self.dest = self.base / "models" / "m"
        self.work = self.base / "fetchwork"

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_repo_url(self):
        self.assertTrue(fetch.repo_url("Xephex/models").endswith("/remote/Xephex/models.git"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(fetch.repo_url("Xephex/workbench-models"),
                             "https://github.com/Xephex/workbench-models.git")

    def test_fetch_file_assembles_and_verifies(self):
        seen = []
        out = fetch.fetch_file(self.f, self.dest, self.work, progress=lambda done, total: seen.append(done))
        self.assertEqual(out.read_bytes(), self.data)
        self.assertTrue((self.dest / "m.gguf.ok").exists())
        self.assertFalse((self.dest / "m.gguf.partial").exists())
        self.assertFalse((self.dest / "m.gguf.fetching").exists())
        self.assertEqual(seen[-1], 10_000)

    def test_second_fetch_is_a_no_op(self):
        fetch.fetch_file(self.f, self.dest, self.work)
        with mock.patch.object(fetch, "_git") as g:
            fetch.fetch_file(self.f, self.dest, self.work)
        g.assert_not_called()

    def test_resume_after_partial_chunk(self):
        """Review focus 1: a crash mid-chunk must not duplicate or corrupt data."""
        real = fetch._append_chunk
        calls = {"n": 0}

        def flaky(bare, chunk, out):
            calls["n"] += 1
            if calls["n"] == 3:
                out.write(b"half-written-garbage")  # simulate a crash in the middle of chunk 3
                raise KeyboardInterrupt
            return real(bare, chunk, out)

        with mock.patch.object(fetch, "_append_chunk", side_effect=flaky):
            with self.assertRaises(KeyboardInterrupt):
                fetch.fetch_file(self.f, self.dest, self.work)
        self.assertFalse((self.dest / "m.gguf").exists())
        self.assertEqual((self.dest / "m.gguf.fetching").read_text().strip(), "2")
        out = fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(out.read_bytes(), self.data)

    def test_sha_mismatch_removes_partial_and_raises(self):
        bad = catalog.ModelFile(self.f.name, "0" * 64, self.f.bytes, self.f.repo, self.f.chunks, "main")
        with self.assertRaisesRegex(fetch.FetchError, "m.gguf"):
            fetch.fetch_file(bad, self.dest, self.work)
        self.assertFalse((self.dest / "m.gguf.partial").exists())
        self.assertFalse((self.dest / "m.gguf").exists())

    def test_unknown_blob_raises_fetch_error(self):
        bad = catalog.ModelFile(self.f.name, self.f.sha256, self.f.bytes, self.f.repo,
                                (catalog.Chunk("1" * 40, self.f.bytes),), "main")
        with self.assertRaises(fetch.FetchError):
            fetch.fetch_file(bad, self.dest, self.work)

    def test_wrong_chunk_size_raises(self):
        c0 = self.f.chunks[0]
        bad_chunks = (catalog.Chunk(c0.blob, c0.bytes + 1),) + self.f.chunks[1:]
        bad = catalog.ModelFile(self.f.name, self.f.sha256, self.f.bytes + 1, self.f.repo, bad_chunks, "main")
        with self.assertRaisesRegex(fetch.FetchError, "size"):
            fetch.fetch_file(bad, self.dest, self.work)

    def test_fetch_model_and_is_installed(self):
        m = catalog.Model("m", ("chat",), "llama.cpp", (self.f,), 1, 0, ("cpu",), "Apache-2.0", "x")
        models_dir = self.base / "models"
        self.assertFalse(fetch.is_installed(m, models_dir))
        fetch.fetch_model(m, models_dir, self.work)
        self.assertTrue(fetch.is_installed(m, models_dir))

    def test_read_only_git_objects_are_removed(self):
        """C1: Windows cannot rmtree read-only files; git makes its objects read-only."""
        real_unlink = os.unlink

        def windows_like_unlink(path, *, dir_fd=None):
            if not os.access(path, os.W_OK, dir_fd=dir_fd):
                raise PermissionError(13, "Access is denied", str(path))
            return real_unlink(path, dir_fd=dir_fd)

        with mock.patch("os.unlink", side_effect=windows_like_unlink):
            out = fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(out.read_bytes(), self.data)
        self.assertEqual([p for p in self.work.iterdir()], [])

    def test_empty_state_file_resumes_from_start(self):
        """I8: power loss can leave an empty .fetching file."""
        self.dest.mkdir(parents=True)
        (self.dest / "m.gguf.fetching").write_text("")
        out = fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(out.read_bytes(), self.data)

    def test_state_beyond_partial_is_capped(self):
        """I8: .fetching says 3 chunks but .partial holds only one; never zero-fill."""
        self.dest.mkdir(parents=True)
        (self.dest / "m.gguf.fetching").write_text("3\n")
        (self.dest / "m.gguf.partial").write_bytes(self.data[:3_000])
        seen = []
        with mock.patch.object(fetch, "_append_chunk", wraps=fetch._append_chunk) as ap:
            out = fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(out.read_bytes(), self.data)
        self.assertEqual(ap.call_count, len(self.f.chunks) - 1)

    def test_half_created_bare_repo_is_rebuilt(self):
        """I8: Ctrl+C between git init and git config leaves a repo with no remote."""
        bare = self.work / "Xephex_models.git"
        bare.mkdir(parents=True)
        import subprocess
        subprocess.run(["git", "init", "-q", "--bare"], cwd=bare, check=True)
        out = fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(out.read_bytes(), self.data)

    def test_renamed_but_no_ok_marker_is_finished_without_refetch(self):
        """I8: a crash between the rename and writing .ok must not force a re-download."""
        fetch.fetch_file(self.f, self.dest, self.work)
        (self.dest / "m.gguf.ok").unlink()
        with mock.patch.object(fetch, "_git") as g:
            out = fetch.fetch_file(self.f, self.dest, self.work)
        g.assert_not_called()
        self.assertEqual(out.read_bytes(), self.data)
        self.assertTrue((self.dest / "m.gguf.ok").exists())

    def test_remaining_bytes_counts_only_missing_chunks(self):
        """I3: a mostly downloaded file needs only its missing chunks."""
        self.dest.mkdir(parents=True)
        (self.dest / "m.gguf.fetching").write_text("2\n")
        (self.dest / "m.gguf.partial").write_bytes(self.data[:6_000])
        self.assertEqual(fetch.remaining_bytes(self.f, self.dest), 4_000)
        fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(fetch.remaining_bytes(self.f, self.dest), 0)

    def test_scratch_peak_is_about_one_chunk(self):
        """I2 / spec A5: the throwaway repo never holds more than one chunk."""
        sizes = []
        real = fetch._append_chunk

        def measure(bare, chunk, out):
            real(bare, chunk, out)
            sizes.append(sum(p.stat().st_size for p in (bare / "objects").rglob("*") if p.is_file()))

        with mock.patch.object(fetch, "_append_chunk", side_effect=measure):
            fetch.fetch_file(self.f, self.dest, self.work)
        self.assertEqual(len(sizes), len(self.f.chunks))
        self.assertLess(max(sizes), 5_500)  # one 3 KB chunk plus pack index; two would be ~8 KB

    def test_remove_files_deletes_partial_state(self):
        """I4: removing a model also removes its half-finished download."""
        self.dest.mkdir(parents=True)
        for suffix in ("", ".ok", ".partial", ".fetching"):
            (self.dest / f"m.gguf{suffix}").write_text("x")
        fetch.remove_file(self.f, self.dest)
        self.assertEqual(list(self.dest.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
