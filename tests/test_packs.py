import unittest

from workbench import hw, packs

GB = 1_000_000_000


def prof(os="windows", arch="x86_64", gpus=(), rosetta=False, wsl=False):
    return hw.Profile(os=os, arch=arch, wsl=wsl, rosetta=rosetta, cpu_name="x", cpu_cores=8, ram_bytes=32 * GB,
                      ram_available_bytes=None, disk_free_bytes=500 * GB, gpus=list(gpus), vulkan_loader=True)


def nvidia(cuda=13010, cc=(12, 0)):
    return hw.Gpu("nvidia", "RTX PRO 2000", 8 * GB, "581.42", cuda, cc, False, "nvml")


class PackTest(unittest.TestCase):
    def test_windows_blackwell(self):
        self.assertEqual(packs.choose_pack(prof(gpus=[nvidia()])), packs.PackChoice("win-x64-cuda12", None))

    def test_windows_no_gpu(self):
        self.assertEqual(packs.choose_pack(prof()).pack, "win-x64-vulkan")

    def test_windows_amd(self):
        amd = hw.Gpu("amd", "RX 7600M", 8 * GB, None, None, None, False, "dxgi")
        self.assertEqual(packs.choose_pack(prof(gpus=[amd])).pack, "win-x64-vulkan")

    def test_old_nvidia_driver_gets_vulkan_with_reason(self):
        """Review focus 4."""
        c = packs.choose_pack(prof(gpus=[nvidia(cuda=12020)]))
        self.assertEqual(c.pack, "win-x64-vulkan")
        self.assertIn("570", c.note)

    def test_pascal_gets_vulkan(self):
        c = packs.choose_pack(prof(gpus=[nvidia(cc=(6, 1))]))
        self.assertEqual(c.pack, "win-x64-vulkan")
        self.assertIn("older than Turing", c.note)

    def test_linux_and_wsl_cuda13(self):
        self.assertEqual(packs.choose_pack(prof(os="linux", gpus=[nvidia()])).pack, "linux-x64-cuda13")
        self.assertEqual(packs.choose_pack(prof(os="linux", wsl=True, gpus=[nvidia()])).pack, "linux-x64-cuda13")

    def test_linux_driver_570_gets_vulkan_and_580_note(self):
        c = packs.choose_pack(prof(os="linux", gpus=[nvidia(cuda=12080)]))
        self.assertEqual(c.pack, "linux-x64-vulkan")
        self.assertIn("580", c.note)

    def test_linux_arm(self):
        self.assertEqual(packs.choose_pack(prof(os="linux", arch="arm64")).pack, "linux-arm64-cpu")

    def test_mac(self):
        apple = hw.Gpu("apple", "M3", None, None, None, None, True, "sysctl")
        self.assertEqual(packs.choose_pack(prof(os="macos", arch="arm64", gpus=[apple])).pack, "macos-arm64-metal")

    def test_unsupported(self):
        self.assertIsNone(packs.choose_pack(prof(os="macos", arch="x86_64")).pack)
        self.assertIsNone(packs.choose_pack(prof(os="windows", arch="arm64")).pack)
        c = packs.choose_pack(prof(os="macos", arch="x86_64", rosetta=True))
        self.assertIsNone(c.pack)
        self.assertIn("arm64", c.note)


if __name__ == "__main__":
    unittest.main()
