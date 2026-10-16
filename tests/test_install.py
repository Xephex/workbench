import json, os, pathlib, shutil, subprocess, sys, tempfile, unittest
from unittest import mock

from tests import wheelfixture
from workbench import install

REPO = pathlib.Path(__file__).resolve().parents[1]
PACK = "linux-x64-vulkan"


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.tmp.name)
        self.root = base / "wb"
        (self.root / "packs" / "pip").mkdir(parents=True)
        shutil.copy(REPO / install.PIP_WHEEL, self.root / install.PIP_WHEEL)
        build = base / "build"
        build.mkdir()
        mk = wheelfixture.make_wheel
        self.big = mk(build, "wbbig", "1.0", extra_files={"wbbig/blob.bin": os.urandom(3000)})
        layers = {
            "core": [mk(build, "wbcore", "1.0", extra_files={"wbcore/zeros.bin": bytes(2_000_000)}), self.big],
            "llama": [mk(build, "wbllama", "1.0", script=True, extra_files={"wbllama/tool.exe": b"MZ fake"})],
            "tracing": [mk(build, "wbtrace", "1.0")],
        }
        for layer, wheels in layers.items():
            d = self.layer(layer)
            d.mkdir(parents=True)
            for w in wheels:
                if w == self.big:
                    wheelfixture.split(w, d, 1500)
                else:
                    shutil.copy(w, d)
            (d / "requirements.lock").write_text("".join(wheelfixture.lock_line(w) for w in wheels))
            (d / "SHA256SUMS").write_text("".join(f"{wheelfixture.sha256(w)}  {w.name}\n" for w in wheels))

    def tearDown(self):
        self.tmp.cleanup()

    def layer(self, name):
        return self.root / "packs" / PACK / name

    def importable(self, module):
        code = (f"import site; site.addsitedir({str(install.site_dir(self.root))!r}); "
                f"import {module}")
        return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True).returncode == 0

    def test_installs_all_layers_and_activates(self):
        self.assertEqual(install.install_pack(self.root, PACK), "installed")
        for module in ("wbcore", "wbbig", "wbllama", "wbtrace"):
            self.assertTrue(self.importable(module), module)
        self.assertEqual(install.installed_pack(self.root), PACK)
        self.assertFalse((self.root / ".workbench" / "wheels").exists())  # joined copies are not kept

    def test_activate(self):
        self.assertFalse(install.activate(self.root))
        install.install_pack(self.root, PACK)
        with mock.patch("site.addsitedir") as add:
            self.assertTrue(install.activate(self.root))
        add.assert_called_once_with(str(install.site_dir(self.root)))

    def test_second_run_is_up_to_date(self):
        install.install_pack(self.root, PACK)
        self.assertEqual(install.install_pack(self.root, PACK), "up to date")

    def test_no_trace_viewer_skips_layer(self):
        install.install_pack(self.root, PACK, with_tracing=False)
        self.assertFalse(self.importable("wbtrace"))
        self.assertTrue(self.importable("wbllama"))
        self.assertEqual(install.install_pack(self.root, PACK, with_tracing=True), "installed")
        self.assertTrue(self.importable("wbtrace"))

    def test_no_exe_left(self):
        install.install_pack(self.root, PACK)
        exes = [p for p in install.site_dir(self.root).rglob("*") if p.suffix.lower() == ".exe"]
        self.assertEqual(exes, [])

    def test_bad_chunk_named(self):
        """Review focus 3."""
        part = self.layer("core") / f"{self.big.name}.part001"
        data = part.read_bytes()
        part.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaisesRegex(install.InstallError, self.big.name):
            install.install_pack(self.root, PACK)
        self.assertFalse(install.site_dir(self.root).exists())

    def test_chunk_gap_named(self):
        (self.layer("core") / f"{self.big.name}.part001").unlink()
        with self.assertRaisesRegex(install.InstallError, self.big.name):
            install.install_pack(self.root, PACK)

    def test_hash_mismatch_installs_nothing(self):
        """Review focus 3."""
        w = self.layer("llama") / "wbllama-1.0-py3-none-any.whl"
        w.write_bytes(w.read_bytes() + b"\0")
        with self.assertRaisesRegex(install.InstallError, "wbllama-1.0-py3-none-any.whl"):
            install.install_pack(self.root, PACK)
        self.assertFalse(install.site_dir(self.root).exists())

    def test_pip_failure_reports_and_keeps_old_site(self):
        """Review focus 1."""
        install.install_pack(self.root, PACK)
        (self.layer("core") / "requirements.lock").write_text(
            "nosuchpkg==1.0 \\\n    --hash=sha256:" + "0" * 64 + "\n")
        with self.assertRaisesRegex(install.InstallError, "nosuchpkg"):
            install.install_pack(self.root, PACK)
        self.assertTrue(self.importable("wbcore"))
        self.assertEqual(install.installed_pack(self.root), PACK)

    def test_leftover_new_site_is_replaced(self):
        """Ctrl+C during pip leaves site.new behind; the next run starts it afresh."""
        junk = self.root / ".workbench" / "site.new" / "junk"
        junk.mkdir(parents=True)
        self.assertEqual(install.install_pack(self.root, PACK), "installed")
        self.assertFalse((self.root / ".workbench" / "site.new").exists())
        self.assertFalse((install.site_dir(self.root) / "junk").exists())

    def test_user_pip_config_ignored(self):
        """Review focus 2."""
        elsewhere = self.root / "elsewhere"
        env = {"PIP_REQUIRE_VIRTUALENV": "1", "PIP_INDEX_URL": "http://127.0.0.1:9/simple",
               "PIP_TARGET": str(elsewhere), "PIP_NO_BINARY": ":all:"}
        with mock.patch.dict(os.environ, env):
            self.assertEqual(install.install_pack(self.root, PACK), "installed")
        self.assertFalse(elsewhere.exists())
        self.assertTrue(self.importable("wbcore"))

    def test_long_path_problem(self):
        """Review focus 4."""
        deep = pathlib.PureWindowsPath("C:/" + "d" * 150 + "/workbench/.workbench/site")
        self.assertIn("shorter folder", install.long_path_problem(deep, enabled=False))
        self.assertIsNone(install.long_path_problem(deep, enabled=True))
        self.assertIsNone(install.long_path_problem(pathlib.PureWindowsPath("C:/wb/.workbench/site"), enabled=False))

    def test_long_path_stops_install(self):
        with mock.patch.object(install, "long_path_problem", return_value="Move it to a shorter folder."):
            with self.assertRaisesRegex(install.InstallError, "shorter folder"):
                install.install_pack(self.root, PACK)

    def test_missing_pack_folder(self):
        with self.assertRaisesRegex(install.InstallError, "not published"):
            install.install_pack(self.root, "macos-arm64-metal")

    def test_not_enough_disk(self):
        with mock.patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(10, 10, 10)):
            with self.assertRaisesRegex(install.InstallError, "disk"):
                install.install_pack(self.root, PACK)

    def force_reinstall(self):
        """Change the pack's digest without changing what pip installs (a comment in a lock)."""
        lock = self.layer("core") / "requirements.lock"
        lock.write_text(lock.read_text() + "# touched\n")

    def test_machine_pip_config_ignored(self):
        """Final review C1: pip still reads machine-wide config files under --isolated."""
        xdg = pathlib.Path(self.tmp.name) / "xdg"
        (xdg / "pip").mkdir(parents=True)
        (xdg / "pip" / "pip.conf").write_text("[global]\nrequire-virtualenv = true\n"
                                               "find-links = http://127.0.0.1:9/links/\n[install]\nuser = true\n")
        appdata = pathlib.Path(self.tmp.name) / "programdata"
        (appdata / "pip").mkdir(parents=True)
        shutil.copy(xdg / "pip" / "pip.conf", appdata / "pip" / "pip.ini")
        with mock.patch.dict(os.environ, {"XDG_CONFIG_DIRS": str(xdg), "ProgramData": str(appdata)}):
            self.assertEqual(install.install_pack(self.root, PACK), "installed")
        self.assertTrue(self.importable("wbcore"))

    def test_activate_puts_site_first(self):
        """Final review I1: the pinned pack must win over packages the user installed elsewhere."""
        install.install_pack(self.root, PACK)
        with mock.patch.object(sys, "path", ["/somewhere/else", *sys.path]):
            install.activate(self.root)
            self.assertEqual(sys.path[0], str(install.site_dir(self.root)))
            self.assertIn("/somewhere/else", sys.path)

    def test_shadowing_module_loses(self):
        install.install_pack(self.root, PACK)
        shadow = pathlib.Path(self.tmp.name) / "shadow"
        (shadow / "wbcore").mkdir(parents=True)
        (shadow / "wbcore" / "__init__.py").write_text("VERSION = 'shadow'\n")
        code = (f"import sys; sys.path.insert(0, {str(REPO)!r}); from workbench import install; import pathlib; "
                f"install.activate(pathlib.Path({str(self.root)!r})); import wbcore; print(wbcore.VERSION)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONPATH=str(shadow)))
        self.assertEqual(r.stdout.strip(), "1.0", r.stderr)

    def test_failed_swap_restores_old_site(self):
        """Final review I2: if the new site cannot be moved into place, the old one comes back."""
        install.install_pack(self.root, PACK)
        self.force_reinstall()
        real = pathlib.Path.replace

        def replace(src, dst):
            if pathlib.Path(src).name == "site.new":
                raise PermissionError(13, "The process cannot access the file")
            return real(src, dst)

        with mock.patch.object(pathlib.Path, "replace", replace), mock.patch("time.sleep"):
            with self.assertRaisesRegex(install.InstallError, "Close any running workbench"):
                install.install_pack(self.root, PACK)
        self.assertTrue(self.importable("wbcore"))
        self.assertFalse((self.root / ".workbench" / "site.old").exists())

    def test_file_error_is_install_error(self):
        """Final review I2: disk full or a locked file is a message, not a traceback."""
        with mock.patch.object(install, "join_chunks", side_effect=OSError(28, "No space left on device")):
            with self.assertRaisesRegex(install.InstallError, "No space left on device"):
                install.install_pack(self.root, PACK)

    def test_disk_check_counts_unpacked_size(self):
        """Final review I3: the 2 MB of zeros in wbcore unpack from a few KB; the check must see 2 MB."""
        work = self.root / ".workbench"
        work.mkdir()
        with mock.patch("shutil.disk_usage", return_value=shutil._ntuple_diskusage(10**9, 10**9 - 1_000_000, 1_000_000)):
            with self.assertRaisesRegex(install.InstallError, "disk"):
                install.install_pack(self.root, PACK)

    def test_pip_temp_files_stay_in_workbench(self):
        """Final review I3: pip stages the install in its temp folder; keep that on the checked drive."""
        env = install._pip_env(self.root)
        tmp = str(self.root / ".workbench" / "tmp")
        for key in ("TMPDIR", "TEMP", "TMP"):
            self.assertEqual(env[key], tmp)
        self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
        self.assertFalse(any(k.startswith("PIP_") and k != "PIP_CONFIG_FILE" for k in env))

    def test_windows_activate_preloads_msvc_runtime(self):
        """Final review I4: llama.cpp's DLLs need msvcp140.dll, which python.org's Python does not ship."""
        install.install_pack(self.root, PACK)
        dll = install.site_dir(self.root) / "msvcp140.dll"
        dll.write_bytes(b"MZ")
        with mock.patch.object(install.sys, "platform", "win32"), mock.patch.object(install, "_load_dll") as load, \
                mock.patch("site.addsitedir"):
            install.activate(self.root)
        load.assert_called_once_with(dll)


class FetchPackTest(unittest.TestCase):
    def test_sparse_checkout_adds_only_this_pack(self):
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            src = d / "src"
            for rel in ("workbench.py", "packs/pip/p.txt", "packs/linux-x64-vulkan/core/x.txt",
                        "packs/linux-x64-vulkan/tracing/t.txt", "packs/win-x64-cuda12/core/y.txt"):
                (src / rel).parent.mkdir(parents=True, exist_ok=True)
                (src / rel).write_text(rel)
            git("init", "-q", "-b", "main", cwd=src)
            git("add", ".", cwd=src)
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x", cwd=src)
            git("config", "uploadpack.allowFilter", "true", cwd=src)
            clone = d / "clone"
            git("clone", "-q", "--filter=blob:none", "--sparse", src.as_uri(), str(clone), cwd=d)
            install.fetch_pack(clone, "linux-x64-vulkan", ["core"])
            self.assertTrue((clone / "packs/linux-x64-vulkan/core/x.txt").exists())
            self.assertTrue((clone / "packs/pip/p.txt").exists())
            self.assertFalse((clone / "packs/linux-x64-vulkan/tracing").exists())
            self.assertFalse((clone / "packs/win-x64-cuda12").exists())

    def test_full_checkout_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            git("init", "-q", cwd=root)
            install.fetch_pack(root, "linux-x64-vulkan", ["core"])  # no sparse checkout: nothing to do


if __name__ == "__main__":
    unittest.main()
