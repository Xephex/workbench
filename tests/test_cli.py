import io, os, pathlib, tempfile, unittest
from unittest import mock

from tests import gitfixture
from workbench import catalog, cli, hw

GB = 1_000_000_000


def fake_profile(disk_free=500 * GB, ram=32 * GB, gpus=()):
    return hw.Profile(os="linux", arch="x86_64", wsl=False, rosetta=False, cpu_name="Test CPU", cpu_cores=8,
                      ram_bytes=ram, ram_available_bytes=None, disk_free_bytes=disk_free,
                      gpus=list(gpus), vulkan_loader=False)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        self.data = os.urandom(5_000)
        (f,) = gitfixture.make_repo(base / "remote", "Xephex/models", {"t.gguf": self.data}, chunk=2_000)
        chunks = ", ".join(f'{{ blob = "{c.blob}", bytes = {c.bytes} }}' for c in f.chunks)
        (base / "catalog.toml").write_text(f'''
[[model]]
id = "tiny"
kinds = ["chat"]
engine = "llama.cpp"
ram_gb = 1
vram_gb = 0
recommended_for = ["cpu"]
license = "Apache-2.0"
source = "x"

[[model.files]]
name = "t.gguf"
role = "main"
sha256 = "{f.sha256}"
bytes = {f.bytes}
repo = "Xephex/models"
chunks = [{chunks}]

[[model]]
id = "huge"
kinds = ["chat"]
engine = "llama.cpp"
ram_gb = 999
vram_gb = 0
recommended_for = ["cpu"]
license = "Apache-2.0"
source = "x"

[[model.files]]
name = "h.gguf"
role = "main"
sha256 = "00"
bytes = 1
repo = "Xephex/models"
chunks = [{{ blob = "ab", bytes = 1 }}]
''')
        self.patches = [
            mock.patch.dict(os.environ, {"WORKBENCH_REPO_BASE": (base / "remote").as_uri()}),
            mock.patch.object(cli, "CATALOG", base / "catalog.toml"),
            mock.patch.object(cli, "MODELS", base / "models"),
            mock.patch.object(cli, "WORK", base / "work"),
            mock.patch.object(hw, "detect", return_value=fake_profile()),
        ]
        for p in self.patches:
            p.start()
        self.base = base

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def run_cli(self, *argv, stdin=""):
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("builtins.input", side_effect=list(stdin.splitlines()) or EOFError):
            code = cli.main(list(argv))
        return code, out.getvalue()

    def test_status_lists_models_with_reasons(self):
        code, out = self.run_cli("status")
        self.assertEqual(code, 0)
        self.assertIn("Tier: cpu", out)
        self.assertIn("Platform pack: linux-x64-vulkan", out)
        self.assertIn("[ ] tiny", out)
        self.assertIn("[-] huge", out)
        self.assertIn("needs 999 GB of RAM", out)

    def test_models_yes_downloads_recommended(self):
        code, out = self.run_cli("models", "--yes")
        self.assertEqual(code, 0, out)
        self.assertEqual((self.base / "models" / "tiny" / "t.gguf").read_bytes(), self.data)
        code, out = self.run_cli("status")
        self.assertIn("[x] tiny", out)

    def test_models_refuses_when_disk_short(self):
        """Review focus 2."""
        with mock.patch.object(hw, "detect", return_value=fake_profile(disk_free=10)):
            code, out = self.run_cli("models", "--add", "tiny")
        self.assertEqual(code, 1)
        self.assertIn("Not enough disk", out)
        self.assertFalse((self.base / "models" / "tiny").exists())

    def test_add_unknown_id(self):
        code, out = self.run_cli("models", "--add", "nope")
        self.assertEqual(code, 2)
        self.assertIn("tiny", out)

    def test_add_unrunnable_refused(self):
        code, out = self.run_cli("models", "--add", "huge")
        self.assertEqual(code, 1)
        self.assertIn("needs 999 GB of RAM", out)

    def test_remove(self):
        self.run_cli("models", "--add", "tiny")
        code, out = self.run_cli("models", "--remove", "tiny")
        self.assertEqual(code, 0)
        self.assertFalse((self.base / "models" / "tiny").exists())

    def test_interactive_default_is_recommended(self):
        code, out = self.run_cli("models", stdin="\n")
        self.assertEqual(code, 0, out)
        self.assertTrue((self.base / "models" / "tiny" / "t.gguf").exists())

    def test_setup_stops_on_unsupported_platform(self):
        mac_intel = hw.Profile(os="macos", arch="x86_64", wsl=False, rosetta=False, cpu_name="i7", cpu_cores=4,
                               ram_bytes=16 * GB, ram_available_bytes=None, disk_free_bytes=100 * GB,
                               gpus=[], vulkan_loader=False)
        with mock.patch.object(hw, "detect", return_value=mac_intel):
            code, out = self.run_cli("setup", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("Intel Macs are not supported", out)

    def test_setup_yes(self):
        code, out = self.run_cli("setup", "--yes")
        self.assertEqual(code, 0, out)
        self.assertIn("Platform pack: linux-x64-vulkan", out)
        self.assertTrue((self.base / "models" / "tiny" / "t.gguf").exists())

    def test_ctrl_c_at_prompt_aborts(self):
        """I5: Ctrl+C must never start the default download."""
        out = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("builtins.input", side_effect=KeyboardInterrupt):
            code = cli.main(["models"])
        self.assertNotEqual(code, 0)
        self.assertFalse((self.base / "models" / "tiny").exists())
        self.assertIn("Cancelled", out.getvalue())

    def test_interactive_unrunnable_pick_explains_why(self):
        """I6: typing a model that cannot run says why (incl. the WSL hint)."""
        code, out = self.run_cli("models", stdin="huge\n")
        self.assertEqual(code, 2)
        self.assertIn("huge: needs 999 GB of RAM", out)

    def test_setup_lists_models_that_cannot_run(self):
        """I6 / spec B2: setup explains what is missing, not only what fits."""
        code, out = self.run_cli("setup", stdin="\n")
        self.assertEqual(code, 0, out)
        self.assertIn("huge", out)
        self.assertIn("needs 999 GB of RAM", out)

    def test_disk_check_counts_only_missing_chunks(self):
        """I3: a nearly finished download must be resumable on a nearly full disk."""
        from workbench import fetch as wfetch
        models = catalog.load(self.base / "catalog.toml")
        tiny = next(m for m in models if m.id == "tiny")
        dest = self.base / "models" / "tiny"
        dest.mkdir(parents=True)
        (dest / "t.gguf.fetching").write_text("2\n")
        (dest / "t.gguf.partial").write_bytes(self.data[:4_000])
        need = cli.plan_downloads([tiny], self.base / "models")
        self.assertEqual(need, 1_000)

    def test_disk_check_includes_scratch_chunk(self):
        """I2: free space must cover the remaining bytes plus one chunk of scratch."""
        with mock.patch.object(hw, "detect", return_value=fake_profile(disk_free=5_500)):
            code, out = self.run_cli("models", "--add", "tiny")
        self.assertEqual(code, 1)
        self.assertIn("Not enough disk", out)

    def test_remove_deletes_partial_download(self):
        """I4: --remove also deletes a half-finished download."""
        dest = self.base / "models" / "tiny"
        dest.mkdir(parents=True)
        (dest / "t.gguf.partial").write_bytes(b"x" * 100)
        (dest / "t.gguf.fetching").write_text("1\n")
        code, out = self.run_cli("models", "--remove", "tiny")
        self.assertEqual(code, 0)
        self.assertFalse(dest.exists())

    def test_os_error_during_download_is_reported(self):
        """C1/I2: an OSError (disk full, file locked) is a message, not a traceback."""
        from workbench import fetch as wfetch
        with mock.patch.object(wfetch, "fetch_model", side_effect=OSError(28, "No space left on device")):
            code, out = self.run_cli("models", "--add", "tiny")
        self.assertEqual(code, 1)
        self.assertIn("No space left on device", out)


if __name__ == "__main__":
    unittest.main()
