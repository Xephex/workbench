"""Which platform pack this machine installs (spec B2, B4)."""

from __future__ import annotations

import dataclasses

from workbench import hw

TURING = (7, 5)
CUDA_MIN = {"windows": (12080, "570"), "linux": (13000, "580")}


@dataclasses.dataclass
class PackChoice:
    pack: str | None
    note: str | None


def _cuda_ok(p: hw.Profile) -> tuple[bool, str | None]:
    nv = [g for g in p.gpus if g.vendor == "nvidia"]
    if not nv:
        return False, None
    g = nv[0]
    if g.compute_capability and g.compute_capability < TURING:
        return False, f"{g.name} is older than Turing, so it runs through Vulkan instead of CUDA."
    need, driver = CUDA_MIN[p.os]
    if not g.cuda_driver or g.cuda_driver < need:
        return False, (f"NVIDIA driver {g.driver or 'unknown'} is too old for CUDA here; update to driver "
                       f"{driver} or newer for full GPU speed. Using Vulkan meanwhile.")
    return True, None


def choose_pack(p: hw.Profile) -> PackChoice:
    if p.rosetta:
        return PackChoice(None, "This Python runs under Rosetta (x86_64). Install the arm64 Python 3.14 "
                                "from python.org and run setup with it.")
    if p.os == "macos":
        if p.arch == "arm64":
            return PackChoice("macos-arm64-metal", None)
        return PackChoice(None, "Intel Macs are not supported yet (several packages have no Python 3.14 build).")
    if p.os == "windows":
        if p.arch != "x86_64":
            return PackChoice(None, "Windows on ARM is not supported yet.")
        ok, note = _cuda_ok(p)
        return PackChoice("win-x64-cuda12" if ok else "win-x64-vulkan", note)
    if p.arch == "arm64":
        return PackChoice("linux-arm64-cpu", None)
    ok, note = _cuda_ok(p)
    return PackChoice("linux-x64-cuda13" if ok else "linux-x64-vulkan", note)
