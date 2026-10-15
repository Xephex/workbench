import os, pathlib, tempfile, unittest
from unittest import mock

from workbench import hw

FIX = pathlib.Path(__file__).parent / "fixtures"
GB = 1_000_000_000


def make_sysfs(root: pathlib.Path, cards: dict) -> None:
    """cards: {"card0": {"vendor": "0x1002", "device": "0x73df", "driver": "amdgpu", "vram": 12*GB}}"""
    for card, info in cards.items():
        dev = root / card / "device"
        dev.mkdir(parents=True)
        (dev / "vendor").write_text(info["vendor"] + "\n")
        (dev / "device").write_text(info["device"] + "\n")
        drv = root / "drivers" / info["driver"]
        drv.mkdir(parents=True, exist_ok=True)
        (dev / "driver").symlink_to(drv)
        if "vram" in info:
            (dev / "mem_info_vram_total").write_text(f"{info['vram']}\n")
    (root / "card0-HDMI-A-1").mkdir()  # connector entry, must be skipped
    (root / "renderD128").mkdir()       # render node, must be skipped


class TierTest(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(hw.tier_for(0, "windows"), "cpu")
        self.assertEqual(hw.tier_for(4 * GB, "linux"), "cpu")
        self.assertEqual(hw.tier_for(int(7.96 * GB), "windows"), "gpu-8")
        self.assertEqual(hw.tier_for(int(11.99 * GB), "linux"), "gpu-16")
        self.assertEqual(hw.tier_for(16 * GB, "macos"), "gpu-16")


class BudgetTest(unittest.TestCase):
    def profile(self, **kw):
        base = dict(os="linux", arch="x86_64", wsl=False, rosetta=False, cpu_name="x", cpu_cores=8,
                    ram_bytes=32 * GB, ram_available_bytes=None, disk_free_bytes=500 * GB, gpus=[],
                    vulkan_loader=False)
        base.update(kw)
        return hw.Profile(**base)

    def test_integrated_gpu_does_not_count(self):
        igpu = hw.Gpu("intel", "Intel UHD 620", 512 * 1024**2, None, None, None, True, "sysfs")
        self.assertEqual(self.profile(gpus=[igpu]).gpu_budget_bytes, 0)
        self.assertEqual(self.profile(gpus=[igpu]).tier, "cpu")

    def test_dedicated_gpu_counts(self):
        dgpu = hw.Gpu("nvidia", "RTX PRO 2000", 8 * GB, "581.42", 13010, (12, 0), False, "nvml")
        self.assertEqual(self.profile(gpus=[dgpu]).tier, "gpu-8")

    def test_apple_uses_two_thirds_of_unified_memory(self):
        apple = hw.Gpu("apple", "Apple M3 Pro", None, None, None, None, True, "sysctl")
        p = self.profile(os="macos", arch="arm64", ram_bytes=36 * GB, gpus=[apple])
        self.assertEqual(p.gpu_budget_bytes, int(36 * GB * 0.66))
        self.assertEqual(p.tier, "gpu-16")
        p = self.profile(os="macos", arch="arm64", ram_bytes=16 * GB, gpus=[apple])
        self.assertEqual(p.tier, "gpu-8")


class LinuxProbeTest(unittest.TestCase):
    def test_read_meminfo(self):
        total, avail = hw.read_meminfo(FIX / "meminfo.txt")
        self.assertEqual(total, 65574956 * 1024)
        self.assertEqual(avail, 38114240 * 1024)

    def test_is_wsl(self):
        self.assertTrue(hw.is_wsl(FIX / "osrelease-wsl.txt"))
        self.assertFalse(hw.is_wsl(FIX / "meminfo.txt"))
        self.assertFalse(hw.is_wsl(FIX / "does-not-exist"))

    def test_sysfs_amd_with_vram(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            make_sysfs(root, {"card0": {"vendor": "0x1002", "device": "0x73df", "driver": "amdgpu", "vram": 12 * GB}})
            gpus = hw.probe_sysfs(root)
        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].vendor, "amd")
        self.assertEqual(gpus[0].vram_bytes, 12 * GB)
        self.assertFalse(gpus[0].integrated)

    def test_sysfs_intel_igpu_has_unknown_vram_and_is_integrated(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            make_sysfs(root, {"card1": {"vendor": "0x8086", "device": "0x5917", "driver": "i915"}})
            gpus = hw.probe_sysfs(root)
        self.assertEqual([(g.vendor, g.vram_bytes, g.integrated) for g in gpus], [("intel", None, True)])

    def test_sysfs_skips_nvidia_left_to_nvml(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            make_sysfs(root, {"card0": {"vendor": "0x10de", "device": "0x2c38", "driver": "nvidia"}})
            self.assertEqual(hw.probe_sysfs(root), [])

    def test_sysfs_missing_root(self):
        self.assertEqual(hw.probe_sysfs(pathlib.Path("/nonexistent/drm")), [])


class DetectTest(unittest.TestCase):
    def test_profile_without_any_gpu_library(self):
        """Review focus 3: a plain laptop with no GPU libraries must give a valid cpu profile."""
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(hw, "probe_nvml", return_value=[]), \
             mock.patch.object(hw, "probe_cuda", return_value=[]), \
             mock.patch.object(hw, "probe_dxgi", return_value=[]), \
             mock.patch.object(hw, "probe_registry", return_value=[]), \
             mock.patch.object(hw, "probe_sysfs", return_value=[]), \
             mock.patch.object(hw, "probe_apple", return_value=[]):
            p = hw.detect(pathlib.Path(d))
        self.assertEqual(p.gpus, [])
        self.assertEqual(p.tier, "cpu")
        self.assertGreater(p.ram_bytes, 0)
        self.assertGreater(p.disk_free_bytes, 0)
        self.assertIn(p.os, ("windows", "linux", "macos"))

    def test_real_probes_never_raise_on_this_machine(self):
        with tempfile.TemporaryDirectory() as d:
            p = hw.detect(pathlib.Path(d))
        self.assertIsInstance(p.gpus, list)

    def test_windows_nvml_result_suppresses_duplicate_nvidia_from_dxgi(self):
        nv = hw.Gpu("nvidia", "RTX PRO 2000", 8 * GB, "581.42", 13010, (12, 0), False, "nvml")
        dx_nv = hw.Gpu("nvidia", "NVIDIA RTX PRO 2000", 8 * GB, None, None, None, False, "dxgi")
        dx_intel = hw.Gpu("intel", "Intel Arc 130T", None, None, None, None, True, "dxgi")
        with tempfile.TemporaryDirectory() as d, \
             mock.patch.object(hw, "probe_nvml", return_value=[nv]), \
             mock.patch.object(hw, "probe_cuda", return_value=[]), \
             mock.patch.object(hw, "probe_dxgi", return_value=[dx_nv, dx_intel]), \
             mock.patch.object(hw, "probe_registry", return_value=[]), \
             mock.patch.object(hw, "_windows_ram", return_value=(64 * GB, 40 * GB)), \
             mock.patch.object(hw, "_cpu_name", return_value="Intel Core Ultra 7 265H"), \
             mock.patch.object(hw, "_vulkan_loader", return_value=True):
            p = hw.detect(pathlib.Path(d), plat="windows")
        self.assertEqual([(g.vendor, g.source) for g in p.gpus], [("nvidia", "nvml"), ("intel", "dxgi")])
        self.assertEqual(p.tier, "gpu-8")
        self.assertEqual(p.os, "windows")

    def test_detect_never_calls_platform_machine(self):
        """I1: platform.machine() can fall back to `cmd /c ver` on Windows when WMI fails."""
        with tempfile.TemporaryDirectory() as d, \
             mock.patch("platform.machine", side_effect=AssertionError("platform.machine called")), \
             mock.patch("platform.uname", side_effect=AssertionError("platform.uname called")):
            p = hw.detect(pathlib.Path(d))
        self.assertIn(p.arch, ("x86_64", "arm64"))

    def test_windows_arch_from_environment(self):
        cases = [({"PROCESSOR_ARCHITECTURE": "AMD64"}, "x86_64"),
                 ({"PROCESSOR_ARCHITECTURE": "ARM64"}, "arm64"),
                 ({"PROCESSOR_ARCHITECTURE": "AMD64", "PROCESSOR_ARCHITEW6432": "ARM64"}, "arm64"),
                 ({"PROCESSOR_ARCHITECTURE": "x86", "PROCESSOR_ARCHITEW6432": "AMD64"}, "x86_64")]
        for env, want in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True), \
                 mock.patch.object(hw, "_windows_native_machine", return_value=None):
                self.assertEqual(hw._arch("windows"), want)


if __name__ == "__main__":
    unittest.main()
