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


if __name__ == "__main__":
    unittest.main()
