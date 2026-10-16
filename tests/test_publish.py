import contextlib, hashlib, http.server, io, json, os, pathlib, subprocess, sys, tempfile, threading, unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import publish_model  # noqa: E402

from tests.test_gguf import gguf_bytes  # noqa: E402
from workbench import catalog, fetch  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "Xephex/workbench-models"
REV = "a" * 40
TRAILER = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def git(*args, cwd) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Source:
    """A Hugging Face stand-in on 127.0.0.1: serves files, honours Range, records each request's Range."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.ranges: list[str | None] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                data = outer.files.get(self.path)
                if data is None:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                outer.ranges.append(rng)
                start = int(rng.removeprefix("bytes=").split("-")[0]) if rng else 0
                if rng:
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                self.wfile.write(data[start:])

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class PublishTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.remote = self.base / "remote" / f"{REPO}.git"
        self.remote.mkdir(parents=True)
        self.initial = self.make_remote(initial=True)
        self.source = Source()
        self.addCleanup(self.source.close)
        for p in (mock.patch.dict(os.environ, {"WORKBENCH_REPO_BASE": (self.base / "remote").as_uri(),
                                               "no_proxy": "*", "NO_PROXY": "*"}),
                  mock.patch.object(publish_model, "HF_BASE", self.source.base),
                  mock.patch.object(publish_model, "CHUNK", 2000),
                  mock.patch.object(publish_model, "RETRY_WAIT", 0)):
            p.start()
            self.addCleanup(p.stop)
        self.main_data = os.urandom(5000)  # 3 parts of 2000, 2000, 1000 bytes
        self.mm_data = os.urandom(1500)    # 1 part
        self.catalog = self.base / "catalog.toml"
        self.catalog.write_text("# test catalog\n", encoding="utf-8")
        self.work = self.base / "work"
        self.spec = self.base / "publish.toml"
        self.models = [self.model()]

    def tearDown(self):
        self.tmp.cleanup()

    # helpers -----------------------------------------------------------------

    def make_remote(self, initial: bool) -> str | None:
        """A bare "GitHub" repo; like Xephex/workbench-models, it starts with an Initial commit holding LICENSE."""
        git("init", "-q", "--bare", "-b", "main", cwd=self.remote)
        git("config", "uploadpack.allowFilter", "true", cwd=self.remote)
        git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=self.remote)
        if not initial:
            return None
        seed = self.base / "seed"
        git("init", "-q", "-b", "main", str(seed), cwd=self.base)
        (seed / "LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())
        git("add", "LICENSE", cwd=seed)
        git("-c", "user.name=x", "-c", "user.email=x@x", "commit", "-qm", "Initial commit", cwd=seed)
        git("push", "-q", self.remote.as_uri(), "main", cwd=seed)
        return git("rev-parse", "HEAD", cwd=seed)

    def model(self, mid="m1", kinds=("chat",), files=None, bad_sha=()):
        files = files or [("m.gguf", "main", self.main_data), ("mmproj.gguf", "mmproj", self.mm_data)]
        for name, _, data in files:
            self.source.files[f"/up/{mid}/resolve/{REV}/{name}"] = data
        return {"id": mid, "kinds": list(kinds), "ram_gb": 2, "vram_gb": 1.5, "recommended_for": ["cpu"],
                "license": "Apache-2.0", "source": f"up/{mid}@{REV}",
                "files": [{"name": n, "role": r, "sha256": "0" * 64 if n in bad_sha else sha(d), "bytes": len(d)}
                          for n, r, d in files]}

    def write_spec(self):
        out = []
        for m in self.models:
            out.append("[[model]]")
            out += [f"{k} = {json.dumps(v)}" for k, v in m.items() if k != "files"]
            for f in m["files"]:
                out.append("[[model.files]]")
                out += [f"{k} = {json.dumps(v)}" for k, v in f.items()]
        self.spec.write_text("\n".join(out) + "\n", encoding="utf-8")

    def run_tool(self, *extra, stdin=""):
        self.write_spec()
        buf = io.StringIO()
        argv = ["--spec", str(self.spec), "--work", str(self.work), "--repo", REPO, "--catalog", str(self.catalog), *extra]
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf), \
                mock.patch.object(sys, "stdin", io.StringIO(stdin)):
            code = publish_model.main(argv)
        return code, buf.getvalue()

    def rgit(self, *args) -> str:
        return git(*args, cwd=self.remote)

    def untouched(self) -> bool:
        return self.commits() == [self.initial]

    def tree(self) -> dict[str, str]:
        out = {}
        for line in self.rgit("ls-tree", "-r", "main").splitlines():
            meta, path = line.split("\t", 1)
            out[path] = meta.split()[2]
        return out

    def commits(self) -> list[str]:
        return self.rgit("rev-list", "--reverse", "main").split()

    def changed(self, commit) -> set[str]:
        return set(self.rgit("diff-tree", "--root", "--no-commit-id", "-r", "--name-only", commit).split())

    ALL = {"LICENSE", "m1/m.gguf.part000", "m1/m.gguf.part001", "m1/m.gguf.part002", "m1/m.gguf.sha256",
           "m1/mmproj.gguf.part000", "m1/mmproj.gguf.sha256", "m1/LICENSE", "m1/SOURCE"}
    MODEL = ALL - {"LICENSE"}

    # tests -------------------------------------------------------------------

    def test_blob_id_matches_git(self):
        p = self.base / "x.bin"
        p.write_bytes(os.urandom(3000))
        self.assertEqual(publish_model.blob_id(p), git("hash-object", str(p), cwd=self.base))

    def test_split_and_slices(self):
        src = self.base / "m.gguf"
        src.write_bytes(self.main_data)
        parts = publish_model.split(src, self.base / "parts")
        self.assertEqual([p.name for p in parts], ["m.gguf.part000", "m.gguf.part001", "m.gguf.part002"])
        self.assertEqual([p.bytes for p in parts], [2000, 2000, 1000])
        self.assertEqual([p.blob for p in parts], [publish_model.blob_id(p.path) for p in parts])
        self.assertEqual(b"".join(p.path.read_bytes() for p in parts), self.main_data)
        self.assertEqual(publish_model.slices(list(range(7)), 3), [[0, 1, 2], [3, 4, 5], [6]])

    def test_download_resumes_with_range(self):
        self.write_spec()
        (m,) = publish_model.load_spec(self.spec)
        dest = self.work / "downloads" / "m1"
        dest.mkdir(parents=True)
        (dest / "m.gguf.download").write_bytes(self.main_data[:1234])
        with contextlib.redirect_stdout(io.StringIO()):
            path = publish_model.download(m, m.files[0], dest)
        self.assertEqual(path.read_bytes(), self.main_data)
        self.assertEqual(self.source.ranges, ["bytes=1234-"])
        self.assertFalse((dest / "m.gguf.download").exists())

    def test_publish_new_model_then_fetch_back(self):
        code, out = self.run_tool()
        self.assertEqual(code, 0, out)
        self.assertIn("published m1", out)
        self.assertIn("fetch-back ok m.gguf", out)
        self.assertIn("fetch-back ok mmproj.gguf", out)
        tree = self.tree()
        self.assertEqual(set(tree), self.ALL)
        (m,) = catalog.load(self.catalog)
        self.assertEqual((m.kinds, m.engine, m.ram_gb, m.vram_gb, m.recommended_for, m.license, m.source),
                         (("chat",), "llama.cpp", 2, 1.5, ("cpu",), "Apache-2.0", f"up/m1@{REV}"))
        f, mm = m.files
        self.assertEqual((f.name, f.role, f.repo, f.sha256, f.bytes), ("m.gguf", "main", REPO, sha(self.main_data), 5000))
        self.assertEqual([c.blob for c in f.chunks], [tree[f"m1/m.gguf.part{i:03d}"] for i in range(3)])
        self.assertEqual([c.bytes for c in f.chunks], [2000, 2000, 1000])
        self.assertEqual((mm.role, [c.blob for c in mm.chunks]), ("mmproj", [tree["m1/mmproj.gguf.part000"]]))
        self.assertEqual(self.rgit("cat-file", "blob", tree["m1/m.gguf.sha256"]) + "\n",
                         f"{sha(self.main_data)}  m.gguf\n")
        self.assertEqual(self.rgit("cat-file", "blob", tree["m1/LICENSE"]),
                         (ROOT / "LICENSE").read_text(encoding="utf-8").strip())
        source = self.rgit("cat-file", "blob", tree["m1/SOURCE"])
        for text in ("up/m1", REV, sha(self.main_data), "5000", "Apache-2.0"):
            self.assertIn(text, source)
        commits = self.commits()
        self.assertEqual(commits[0], self.initial)
        self.assertEqual(self.rgit("rev-parse", f"{commits[1]}^"), self.initial)  # built on the existing main
        for c in commits[1:]:
            self.assertEqual(self.rgit("log", "-1", "--format=%an <%ae>|%cn <%ce>", c),
                             "Xephex <Xephex@users.noreply.github.com>|Xephex <Xephex@users.noreply.github.com>")
            self.assertTrue(self.rgit("log", "-1", "--format=%B", c).endswith(TRAILER))
        got = fetch.fetch_file(f, self.base / "check", self.base / "checkwork")
        self.assertEqual(got.read_bytes(), self.main_data)
        # parts deleted after the verified push; downloads and fetch-back copies removed once published
        self.assertFalse((self.work / "parts" / "m1").exists())
        self.assertFalse((self.work / "downloads" / "m1").exists())
        self.assertFalse((self.work / "fetchback" / "m1").exists())

    def test_second_run_changes_nothing(self):
        self.assertEqual(self.run_tool()[0], 0)
        before, requests, text = self.commits(), len(self.source.ranges), self.catalog.read_text()
        code, out = self.run_tool()
        self.assertEqual(code, 0, out)
        self.assertIn("already published m1", out)
        self.assertEqual(self.commits(), before)
        self.assertEqual(len(self.source.ranges), requests)  # nothing downloaded again
        self.assertEqual(self.catalog.read_text(), text)

    def test_rerun_skips_pushed_slices(self):
        """Review focus 5: an interrupted run continues from what the remote already has."""
        real, calls = publish_model.push, []

        def interrupted(*args, **kw):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt  # network gone / Ctrl+C during the second slice
            return real(*args, **kw)

        with mock.patch.object(publish_model, "MAX_FILES", 2):
            with mock.patch.object(publish_model, "push", side_effect=interrupted):
                code, out = self.run_tool()
            self.assertEqual(code, 130, out)
            initial, first = self.commits()
            self.assertEqual(initial, self.initial)
            pushed = self.changed(first)
            self.assertEqual(len(pushed), 2)
            requests = len(self.source.ranges)
            code, out = self.run_tool()
        self.assertEqual(code, 0, out)
        commits = self.commits()
        self.assertEqual(commits[:2], [self.initial, first])
        self.assertEqual(len(commits), 2 + 3)  # 6 files left, 2 per slice
        for c in commits[2:]:
            self.assertFalse(self.changed(c) & pushed, "a pushed file was committed again")
        self.assertEqual(set().union(*(self.changed(c) for c in commits[1:])), self.MODEL)
        self.assertEqual(len(self.source.ranges), requests)  # the kept download was reused
        self.assertIn("published m1", out)

    def test_rejected_push_rebases_on_remote(self):
        """Review focus 5: the remote moved; rebuild on the new head and push again, never forcing."""
        real, calls, other = publish_model.push, [], []

        def moved(*args, **kw):
            calls.append(1)
            if len(calls) == 2:  # someone else pushes to main between our first and second slice
                clone = self.base / "other"
                git("clone", "-q", self.remote.as_uri(), str(clone), cwd=self.base)
                (clone / "NOTE.md").write_text("unrelated change\n")
                git("add", "NOTE.md", cwd=clone)
                git("-c", "user.name=o", "-c", "user.email=o@o", "commit", "-qm", "other change", cwd=clone)
                git("push", "-q", "origin", "HEAD:main", cwd=clone)
                other.append(git("rev-parse", "HEAD", cwd=clone))
            return real(*args, **kw)

        with mock.patch.object(publish_model, "MAX_FILES", 4), \
                mock.patch.object(publish_model, "push", side_effect=moved):
            code, out = self.run_tool()
        self.assertEqual(code, 0, out)
        self.assertEqual(len(calls), 3)  # the second slice was rejected once, then rebuilt
        self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", other[0], "main"],
                                        cwd=self.remote).returncode, 0)
        self.assertEqual(self.rgit("rev-list", "--min-parents=2", "main"), "")
        tree = self.tree()
        self.assertEqual(set(tree), self.ALL | {"NOTE.md"})
        (m,) = catalog.load(self.catalog)
        self.assertEqual([c.blob for c in m.files[0].chunks], [tree[f"m1/m.gguf.part{i:03d}"] for i in range(3)])
        self.assertIn("fetch-back ok m.gguf", out)

    def test_push_error_after_remote_got_it_is_not_repeated(self):
        """Review focus 5: the connection drops after GitHub took the push; do not commit it twice."""
        real, calls = publish_model.push, []

        def dropped(*args, **kw):
            real(*args, **kw)
            calls.append(1)
            if len(calls) == 1:
                raise publish_model.PushError("connection reset by peer")

        with mock.patch.object(publish_model, "push", side_effect=dropped):
            code, out = self.run_tool()
        self.assertEqual(code, 0, out)
        self.assertEqual(len(self.commits()), 2)
        self.assertEqual(set(self.tree()), self.ALL)

    def test_empty_remote_gets_a_first_commit(self):
        self.remote = self.base / "remote" / "Xephex" / "empty.git"
        self.remote.mkdir(parents=True)
        self.make_remote(initial=False)
        code, out = self.run_tool("--repo", "Xephex/empty")
        self.assertEqual(code, 0, out)
        self.assertEqual(set(self.tree()), self.MODEL)
        self.assertEqual(len(self.commits()), 1)
        self.assertEqual(catalog.load(self.catalog)[0].files[0].repo, "Xephex/empty")

    def test_reranker_without_head_not_published(self):
        self.models = [self.model("r1", ("rerank",), [("r.gguf", "main", gguf_bytes(pool=4, cls=False) + os.urandom(3000))])]
        code, out = self.run_tool()
        self.assertEqual(code, 1, out)
        self.assertIn("cls.output.weight", out)
        self.assertTrue(self.untouched())
        self.assertEqual(self.catalog.read_text(), "# test catalog\n")

    def test_reranker_with_head_published(self):
        self.models = [self.model("r1", ("rerank",), [("r.gguf", "main", gguf_bytes(pool=4, cls=True) + os.urandom(3000))])]
        code, out = self.run_tool()
        self.assertEqual(code, 0, out)
        self.assertIn("published r1", out)

    def test_sha_mismatch_stops_before_any_commit(self):
        self.models = [self.model(bad_sha=("mmproj.gguf",))]
        code, out = self.run_tool()
        self.assertEqual(code, 1, out)
        self.assertIn("mmproj.gguf", out)
        self.assertTrue(self.untouched())
        dl = self.work / "downloads" / "m1"
        self.assertFalse((dl / "mmproj.gguf").exists() or (dl / "mmproj.gguf.download").exists())
        self.assertEqual(self.catalog.read_text(), "# test catalog\n")

    def test_token_only_reaches_push(self):
        token, calls, real_run = "s3cret-tok", [], subprocess.run

        def spy(args, *a, **kw):
            calls.append((list(args), dict(kw.get("env") or {})))
            return real_run(args, *a, **kw)

        with mock.patch.object(subprocess, "run", side_effect=spy):
            code, out = self.run_tool("--token-stdin", stdin=token + "\n")
        self.assertEqual(code, 0, out)
        self.assertNotIn(token, out)
        self.assertNotIn("WB_PUSH_TOKEN", os.environ)
        pushes = [(args, env) for args, env in calls if "push" in args]
        self.assertTrue(pushes)
        for args, env in calls:
            self.assertFalse(any(token in str(x) for x in args), args)
            if (args, env) in pushes:
                self.assertEqual(env["WB_PUSH_TOKEN"], token)
                self.assertIn("credential.helper=", args)
            else:
                self.assertNotIn("WB_PUSH_TOKEN", env)

    def test_token_stdin_empty_is_refused(self):
        code, out = self.run_tool("--token-stdin", stdin="\n")
        self.assertEqual(code, 2)
        self.assertIn("token", out)
        self.assertTrue(self.untouched())

    def test_spec_rejects_unsafe_names_and_unpinned_sources(self):
        bad = [dict(self.model(), id="../x"), dict(self.model(), source="up/m1@main")]
        for m in bad:
            self.models = [m]
            self.write_spec()
            with self.assertRaises(publish_model.PublishError):
                publish_model.load_spec(self.spec)

    def test_real_spec(self):
        models = publish_model.load_spec(ROOT / "tools" / "publish.toml")
        self.assertEqual([m.id for m in models], ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3-embed-4b", "qwen3-embed-8b",
                                                  "qwen3-rerank-0.6b", "qwen3-rerank-4b", "qwen3-rerank-8b"])
        self.assertEqual(sum(len(m.files) for m in models), 9)
        self.assertEqual(sum(f.bytes for m in models for f in m.files), 17_672_772_128)
        by_id = {m.id: m for m in models}
        self.assertEqual(by_id["qwen3-rerank-0.6b"].source,
                         "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF@a02f48bb4f057028298c21fa033da2b30d7742d5")
        self.assertEqual(by_id["qwen3-rerank-0.6b"].recommended_for, ("cpu", "gpu-8"))
        self.assertEqual([f.role for f in by_id["qwen3.5-2b"].files], ["main", "mmproj"])


if __name__ == "__main__":
    unittest.main()
