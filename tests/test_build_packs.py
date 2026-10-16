import os, pathlib, shutil, subprocess, sys, tempfile, unittest
from unittest import mock

from tests import wheelfixture
from workbench import install

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import build_packs  # noqa: E402

HAVE_UV = shutil.which("uv") is not None
H = "a" * 64


def pin(name, version="1.0", hashes=(H,)):
    return build_packs.Pin(name, version, list(hashes))


class PureTest(unittest.TestCase):
    def test_read_lock_parses_hashes_and_normalises_names(self):
        text = ("# header\nFoo_Bar==1.0 \\\n    --hash=sha256:" + "a" * 64 + " \\\n    --hash=sha256:" + "c" * 64
                + "\n    # via x\nb==2.0+cu128 \\\n    --hash=sha256:" + "b" * 64 + "\n")
        pins = build_packs.read_lock(text)
        self.assertEqual(sorted(pins), ["b", "foo-bar"])
        self.assertEqual(pins["foo-bar"].hashes, ["a" * 64, "c" * 64])
        self.assertEqual(pins["b"].version, "2.0+cu128")

    def test_read_lock_rejects_unhashed_pin(self):
        with self.assertRaises(build_packs.BuildError):
            build_packs.read_lock("a==1.0\n")

    def test_assign_layers_first_layer_wins(self):
        locks = {"core": {"numpy": pin("numpy")},
                 "llama": {"numpy": pin("numpy"), "llama-cpp-python": pin("llama-cpp-python")},
                 "tracing": {"numpy": pin("numpy"), "arize-phoenix": pin("arize-phoenix")}}
        self.assertEqual(build_packs.assign_layers(locks),
                         {"numpy": "core", "llama-cpp-python": "llama", "arize-phoenix": "tracing"})

    def test_split_big_roundtrips_through_install_join(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            data = os.urandom(2500)
            (d / "x-1.0-py3-none-any.whl").write_bytes(data)
            (d / "small-1.0-py3-none-any.whl").write_bytes(b"s")
            with mock.patch.object(build_packs, "SPLIT_BYTES", 1000):
                build_packs.split_big(d)
            self.assertFalse((d / "x-1.0-py3-none-any.whl").exists())
            self.assertEqual(len(list(d.glob("x-1.0-py3-none-any.whl.part*"))), 3)
            self.assertTrue((d / "small-1.0-py3-none-any.whl").exists())
            install.join_chunks(d, d / "joined")
            self.assertEqual((d / "joined" / "x-1.0-py3-none-any.whl").read_bytes(), data)

    def test_every_pack_has_a_target_and_llama_input(self):
        for pack in ("win-x64-cuda12", "win-x64-vulkan", "linux-x64-cuda13", "linux-x64-vulkan", "linux-arm64-cpu"):
            self.assertIn(pack, build_packs.PACKS)
            self.assertTrue((REPO / "packs" / "requirements" / f"llama-{pack}.in").is_file(), pack)

    def test_linux_platform_list_includes_plain_linux_tag(self):
        plats = build_packs.PACKS["linux-x64-cuda13"].pip_platforms
        self.assertIn("linux_x86_64", plats)          # JamePeng's CUDA wheels are tagged linux_x86_64
        self.assertIn("manylinux_2_34_x86_64", plats)
        self.assertIn("manylinux2014_x86_64", plats)
        self.assertNotIn("manylinux_2_35_x86_64", plats)


@unittest.skipUnless(HAVE_UV, "uv is not installed")
class BuildTest(unittest.TestCase):
    """End to end on fixture wheels: lock, fetch, verify, then install with workbench.install."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        idx = base / "index"
        idx.mkdir()
        mk = wheelfixture.make_wheel
        mk(idx, "wbcore", "1.0", requires=("wbshared>=1",))
        mk(idx, "wbshared", "1.0")
        mk(idx, "wbshared", "2.0")
        mk(idx, "wbwin", "1.0")
        mk(idx, "wbllama", "1.0", requires=("wbshared", 'wbwin ; sys_platform == "win32"'),
           extra_files={"wbllama/blob.bin": os.urandom(3000)})
        mk(idx, "wbmonty", "1.0")
        mk(idx, "wbtrace", "1.0", requires=("wbshared<2", "wbmonty"))
        req = base / "req"
        req.mkdir()
        (req / "core.in").write_text("wbcore\n")
        (req / "tracing.in").write_text("wbtrace\n")
        (req / "overrides.txt").write_text('wbmonty ; sys_platform != "win32"\n')
        for pack in ("linux-x64-vulkan", "win-x64-vulkan"):
            (req / f"llama-{pack}.in").write_text("wbllama==1.0\n")
        self.root = base / "wb"
        (self.root / "packs" / "pip").mkdir(parents=True)
        shutil.copy(REPO / install.PIP_WHEEL, self.root / install.PIP_WHEEL)
        self.base, self.idx, self.req = base, idx, req

    def tearDown(self):
        self.tmp.cleanup()

    def ctx(self, pack):
        return build_packs.Ctx(pack=pack, target=build_packs.PACKS[pack], req_dir=self.req,
                               out_dir=self.root / "packs" / pack, pip_wheel=self.root / install.PIP_WHEEL,
                               find_links=[self.idx], index=False, staging=self.base / "staging" / pack)

    def importable_version(self, module):
        code = (f"import site; site.addsitedir({str(install.site_dir(self.root))!r}); "
                f"import {module}; print({module}.VERSION)")
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    def test_build_verify_install_linux(self):
        c = self.ctx("linux-x64-vulkan")
        with mock.patch.object(build_packs, "SPLIT_BYTES", 1000):
            build_packs.lock(c)
            build_packs.fetch(c)
        build_packs.verify(c)
        core = build_packs.read_lock((c.out_dir / "core" / "requirements.lock").read_text())
        self.assertEqual(core["wbshared"].version, "1.0")  # constrained by tracing's wbshared<2
        self.assertTrue(list((c.out_dir / "llama").glob("wbllama-1.0-py3-none-any.whl.part*")))
        self.assertFalse((c.out_dir / "tracing" / "wbshared-1.0-py3-none-any.whl").exists())  # stored once, in core
        self.assertEqual(install.install_pack(self.root, "linux-x64-vulkan"), "installed")
        self.assertEqual(self.importable_version("wbshared"), "1.0")
        self.assertEqual(self.importable_version("wbmonty"), "1.0")
        self.assertIsNone(self.importable_version("wbwin"))

    def test_windows_locks_drop_monty_and_keep_windows_deps(self):
        c = self.ctx("win-x64-vulkan")
        build_packs.lock(c)
        locks = {layer: (c.out_dir / layer / "requirements.lock").read_text() for layer in install.LAYERS}
        self.assertNotIn("wbmonty", "".join(locks.values()))
        self.assertIn("wbwin==1.0", locks["llama"])

    def test_verify_catches_missing_wheel(self):
        c = self.ctx("linux-x64-vulkan")
        build_packs.lock(c)
        build_packs.fetch(c)
        (c.out_dir / "core" / "wbcore-1.0-py3-none-any.whl").unlink()
        sums = c.out_dir / "core" / "SHA256SUMS"
        sums.write_text("".join(x + "\n" for x in sums.read_text().splitlines() if "wbcore" not in x))
        with self.assertRaisesRegex(build_packs.BuildError, "wbcore"):
            build_packs.verify(c)

    def test_verify_catches_monty_in_windows_lock(self):
        c = self.ctx("win-x64-vulkan")
        build_packs.lock(c)
        build_packs.fetch(c)
        lock = c.out_dir / "tracing" / "requirements.lock"
        lock.write_text(lock.read_text() + "wbmonty==1.0 \\\n    --hash=sha256:" + "0" * 64 + "\n")
        with self.assertRaisesRegex(build_packs.BuildError, "monty"):
            build_packs.verify(c)


if __name__ == "__main__":
    unittest.main()
