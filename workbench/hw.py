"""Hardware profile, using only the standard library and ctypes.

Never starts a process. Every probe is optional: a probe that cannot run
returns [] and detection carries on.
"""

from __future__ import annotations

import ctypes, ctypes.util, dataclasses, os, pathlib, shutil, struct, sys

GB = 1_000_000_000
APPLE_GPU_SHARE = 0.66
TIER_16 = int(11.5 * GB)
TIER_8 = int(5.5 * GB)
PCI_VENDORS = {0x10DE: "nvidia", 0x1002: "amd", 0x8086: "intel"}


@dataclasses.dataclass
class Gpu:
    vendor: str
    name: str
    vram_bytes: int | None
    driver: str | None
    cuda_driver: int | None
    compute_capability: tuple[int, int] | None
    integrated: bool
    source: str


@dataclasses.dataclass
class Profile:
    os: str
    arch: str
    wsl: bool
    rosetta: bool
    cpu_name: str
    cpu_cores: int
    ram_bytes: int
    ram_available_bytes: int | None
    disk_free_bytes: int
    gpus: list[Gpu]
    vulkan_loader: bool

    @property
    def gpu_budget_bytes(self) -> int:
        best = 0
        for g in self.gpus:
            if g.vendor == "apple":
                best = max(best, int(self.ram_bytes * APPLE_GPU_SHARE))
            elif not g.integrated and g.vram_bytes:
                best = max(best, g.vram_bytes)
        return best

    @property
    def tier(self) -> str:
        return tier_for(self.gpu_budget_bytes, self.os)


def tier_for(gpu_budget_bytes: int, os: str) -> str:
    if gpu_budget_bytes >= TIER_16:
        return "gpu-16"
    if gpu_budget_bytes >= TIER_8:
        return "gpu-8"
    return "cpu"


# ---------- Linux files ----------

def read_meminfo(path: pathlib.Path = pathlib.Path("/proc/meminfo")) -> tuple[int, int | None]:
    total = avail = None
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        if key == "MemTotal":
            total = int(rest.split()[0]) * 1024
        elif key == "MemAvailable":
            avail = int(rest.split()[0]) * 1024
    return total or 0, avail


def is_wsl(osrelease: pathlib.Path = pathlib.Path("/proc/sys/kernel/osrelease")) -> bool:
    if os.environ.get("WSL_DISTRO_NAME") and osrelease == pathlib.Path("/proc/sys/kernel/osrelease"):
        return True
    try:
        return "microsoft" in osrelease.read_text().lower()
    except OSError:
        return False


def probe_sysfs(root: pathlib.Path = pathlib.Path("/sys/class/drm")) -> list[Gpu]:
    """AMD and Intel GPUs from /sys/class/drm. NVIDIA is left to NVML."""
    gpus = []
    try:
        cards = sorted(p for p in root.iterdir() if p.name.startswith("card") and p.name[4:].isdigit())
    except OSError:
        return []
    for card in cards:
        dev = card / "device"
        try:
            vendor_id = int((dev / "vendor").read_text().strip(), 16)
        except (OSError, ValueError):
            continue
        vendor = PCI_VENDORS.get(vendor_id, "other")
        if vendor == "nvidia":
            continue
        try:
            driver = os.path.basename(os.readlink(dev / "driver"))
        except OSError:
            driver = None
        vram = None
        try:
            vram = int((dev / "mem_info_vram_total").read_text().strip())
        except (OSError, ValueError):
            pass
        # No VRAM figure (Intel) or a small BIOS carve-out (AMD APU): treat as integrated.
        integrated = vram is None or vram < 2 * GB
        gpus.append(Gpu(vendor, f"{vendor} {card.name}", vram, driver, None, None, integrated, "sysfs"))
    return gpus


# ---------- NVIDIA (Windows and Linux) ----------

class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_uint64), ("free", ctypes.c_uint64), ("used", ctypes.c_uint64)]


def _load_nvml():
    if sys.platform == "win32":
        cands = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "System32", "nvml.dll"),
                 os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                              "NVIDIA Corporation", "NVSMI", "nvml.dll")]
    else:
        cands = ["libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1"]
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    return None


def probe_nvml() -> list[Gpu]:
    lib = _load_nvml()
    if lib is None:
        return []
    try:
        lib.nvmlInit_v2.restype = ctypes.c_int
        if lib.nvmlInit_v2() != 0:
            return []
        try:
            buf = ctypes.create_string_buffer(96)
            lib.nvmlSystemGetDriverVersion.argtypes = [ctypes.c_char_p, ctypes.c_uint]
            driver = buf.value.decode() if lib.nvmlSystemGetDriverVersion(buf, 96) == 0 else None
            cuda = ctypes.c_int(0)
            lib.nvmlSystemGetCudaDriverVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
            cuda_driver = cuda.value if lib.nvmlSystemGetCudaDriverVersion(ctypes.byref(cuda)) == 0 else None
            count = ctypes.c_uint(0)
            lib.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
            if lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
                return []
            lib.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
            lib.nvmlDeviceGetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint]
            lib.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_NvmlMemory)]
            lib.nvmlDeviceGetCudaComputeCapability.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            gpus = []
            for i in range(count.value):
                h = ctypes.c_void_p()
                if lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)) != 0:
                    continue
                name = ctypes.create_string_buffer(96)
                lib.nvmlDeviceGetName(h, name, 96)
                mem = _NvmlMemory()
                vram = mem.total if lib.nvmlDeviceGetMemoryInfo(h, ctypes.byref(mem)) == 0 else None
                ma, mi = ctypes.c_int(0), ctypes.c_int(0)
                cc = (ma.value, mi.value) if lib.nvmlDeviceGetCudaComputeCapability(
                    h, ctypes.byref(ma), ctypes.byref(mi)) == 0 else None
                gpus.append(Gpu("nvidia", name.value.decode(errors="replace"), vram, driver,
                                cuda_driver, cc, False, "nvml"))
            return gpus
        finally:
            lib.nvmlShutdown()
    except (AttributeError, OSError):
        return []


def probe_cuda() -> list[Gpu]:
    """Fallback when NVML is missing but the CUDA driver library exists."""
    names = ["nvcuda.dll"] if sys.platform == "win32" else ["libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"]
    lib = None
    for n in names:
        try:
            lib = ctypes.CDLL(n)
            break
        except OSError:
            continue
    if lib is None:
        return []
    try:
        if lib.cuInit(0) != 0:
            return []
        ver, count = ctypes.c_int(0), ctypes.c_int(0)
        lib.cuDriverGetVersion(ctypes.byref(ver))
        if lib.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return []
        lib.cuDeviceTotalMem_v2.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        gpus = []
        for i in range(count.value):
            dev = ctypes.c_int(0)
            if lib.cuDeviceGet(ctypes.byref(dev), i) != 0:
                continue
            name = ctypes.create_string_buffer(256)
            lib.cuDeviceGetName(name, 256, dev)
            total = ctypes.c_size_t(0)
            vram = total.value if lib.cuDeviceTotalMem_v2(ctypes.byref(total), dev) == 0 else None
            ma, mi = ctypes.c_int(0), ctypes.c_int(0)
            lib.cuDeviceGetAttribute(ctypes.byref(ma), 75, dev)
            lib.cuDeviceGetAttribute(ctypes.byref(mi), 76, dev)
            gpus.append(Gpu("nvidia", name.value.decode(errors="replace"), vram, None, ver.value,
                            (ma.value, mi.value), False, "cuda"))
        return gpus
    except (AttributeError, OSError):
        return []


# ---------- Windows: DXGI and registry ----------

class _Guid(ctypes.Structure):
    _fields_ = [("d1", ctypes.c_uint32), ("d2", ctypes.c_uint16), ("d3", ctypes.c_uint16),
                ("d4", ctypes.c_uint8 * 8)]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint32),
                ("DeviceId", ctypes.c_uint32), ("SubSysId", ctypes.c_uint32), ("Revision", ctypes.c_uint32),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", ctypes.c_uint64),
                ("Flags", ctypes.c_uint32)]


def _vtable_call(obj, slot, restype, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[slot])


def probe_dxgi() -> list[Gpu]:
    if sys.platform != "win32":
        return []
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        iid = _Guid(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_uint8 * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        factory = ctypes.c_void_p()
        dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p)]
        if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return []
        gpus = []
        enum = _vtable_call(factory, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum(factory, i, ctypes.byref(adapter)) != 0:
                break
            desc = _AdapterDesc1()
            get_desc = _vtable_call(adapter, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc1))
            if get_desc(adapter, ctypes.byref(desc)) == 0 and not desc.Flags & 2:
                vendor = PCI_VENDORS.get(desc.VendorId, "other")
                vram = desc.DedicatedVideoMemory or None
                gpus.append(Gpu(vendor, desc.Description, vram, None, None, None,
                                vram is None or vram < 2 * GB, "dxgi"))
            _vtable_call(adapter, 2, ctypes.c_ulong)(adapter)
            i += 1
        _vtable_call(factory, 2, ctypes.c_ulong)(factory)
        return gpus
    except (OSError, AttributeError, ValueError):
        return []


def probe_registry() -> list[Gpu]:
    if sys.platform != "win32":
        return []
    try:
        import winreg
    except ImportError:
        return []
    key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key)
    except OSError:
        return []
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as k:
                    def val(name):
                        try:
                            return winreg.QueryValueEx(k, name)[0]
                        except OSError:
                            return None
                    desc = val("DriverDesc") or "unknown"
                    match = (val("MatchingDeviceId") or "").lower()
                    mem = val("HardwareInformation.qwMemorySize")
                    if isinstance(mem, bytes):
                        mem = int.from_bytes(mem, "little")
                    vendor = "other"
                    for vid, name in PCI_VENDORS.items():
                        if f"ven_{vid:04x}" in match:
                            vendor = name
                    vram = mem if isinstance(mem, int) and mem > 0 else None
                    gpus.append(Gpu(vendor, desc, vram, val("DriverVersion"), None, None,
                                    vram is None or vram < 2 * GB, "registry"))
            except OSError:
                continue
    return gpus


def _windows_ram() -> tuple[int, int | None]:
    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    m = MemStatus()
    m.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys, m.ullAvailPhys


# ---------- macOS ----------

def _sysctl(name: str, as_int: bool):
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    libc.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                  ctypes.c_void_p, ctypes.c_size_t]
    size = ctypes.c_size_t(0)
    if libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
        return None
    if as_int:
        return int.from_bytes(buf.raw[: size.value], "little")
    return buf.value.decode(errors="replace")


def probe_apple() -> list[Gpu]:
    if sys.platform != "darwin":
        return []
    try:
        if _sysctl("hw.optional.arm64", True) != 1:
            return []
        name = _sysctl("machdep.cpu.brand_string", False) or _sysctl("hw.model", False) or "Apple Silicon"
        return [Gpu("apple", name, None, None, None, None, True, "sysctl")]
    except OSError:
        return []


# ---------- detect ----------

def _vulkan_loader() -> bool:
    name = "vulkan-1.dll" if sys.platform == "win32" else ("libvulkan.so.1" if sys.platform != "darwin" else None)
    if not name:
        return False
    try:
        ctypes.CDLL(name)
        return True
    except OSError:
        return False


def _cpu_name(plat: str) -> str:
    """CPU model name without starting a process (platform.processor() may run uname or sysctl)."""
    if plat == "windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except (ImportError, OSError):
            return os.environ.get("PROCESSOR_IDENTIFIER", "unknown CPU")
    if plat == "macos":
        try:
            return _sysctl("machdep.cpu.brand_string", False) or "Apple Silicon"
        except OSError:
            return "Apple Silicon"
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(("model name", "Model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


_IMAGE_FILE_MACHINE = {0x8664: "x86_64", 0xAA64: "arm64"}


def _windows_native_machine() -> str | None:
    """Native CPU type via IsWow64Process2 (Windows 10 1709+); sees through x64 emulation on ARM."""
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = k32.IsWow64Process2
    except (AttributeError, OSError):
        return None
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
    fn.restype = ctypes.c_int
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    process, native = ctypes.c_ushort(), ctypes.c_ushort()
    if not fn(k32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)):
        return None
    return _IMAGE_FILE_MACHINE.get(native.value)


def _arch(plat: str) -> str:
    """CPU architecture without the platform module (platform.machine() can start `cmd /c ver`)."""
    if plat == "windows":
        native = _windows_native_machine()
        if native:
            return native
        name = os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")
        return "arm64" if name.upper() == "ARM64" else "x86_64"
    machine = os.uname().machine.lower() if hasattr(os, "uname") else ""
    return "arm64" if machine in ("arm64", "aarch64") else "x86_64"


def detect(models_dir: pathlib.Path, plat: str | None = None) -> Profile:
    plat = plat or {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    arch = _arch(plat)
    rosetta = False
    cpu_name = _cpu_name(plat)
    if plat == "windows":
        ram, avail = _windows_ram()
        gpus = probe_nvml() or probe_cuda()
        others = [g for g in (probe_dxgi() or probe_registry()) if not (gpus and g.vendor == "nvidia")]
        gpus = gpus + others
    elif plat == "macos":
        ram = _sysctl("hw.memsize", True) or 0
        avail = None
        rosetta = _sysctl("sysctl.proc_translated", True) == 1
        gpus = probe_apple()
    else:
        ram, avail = read_meminfo()
        gpus = (probe_nvml() or probe_cuda()) + probe_sysfs()
    probe_dir = models_dir if models_dir.exists() else models_dir.parent
    while not probe_dir.exists() and probe_dir != probe_dir.parent:
        probe_dir = probe_dir.parent
    return Profile(
        os=plat, arch=arch, wsl=plat == "linux" and is_wsl(), rosetta=rosetta,
        cpu_name=cpu_name, cpu_cores=physical_cores(plat),
        ram_bytes=ram, ram_available_bytes=avail,
        disk_free_bytes=shutil.disk_usage(probe_dir).free,
        gpus=gpus, vulkan_loader=_vulkan_loader(),
    )


def _count_core_records(data: bytes) -> int:
    offset = count = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError('Truncated processor topology')
        relation, size = struct.unpack_from('<II', data, offset)
        if size < 8 or offset + size > len(data):
            raise ValueError('Invalid processor topology')
        count += relation == 0
        offset += size
    return count

def physical_cores(plat=None, sysfs=pathlib.Path('/sys/devices/system/cpu')) -> int:
    plat = plat or ('windows' if sys.platform == 'win32' else 'linux')
    try:
        if plat == 'windows':
            fn = ctypes.WinDLL('kernel32', use_last_error=True).GetLogicalProcessorInformationEx
            fn.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            fn.restype = ctypes.c_int
            size = ctypes.c_uint32()
            fn(0, None, ctypes.byref(size))
            if not size.value:
                raise OSError('No processor topology')
            buf = ctypes.create_string_buffer(size.value)
            if not fn(0, buf, ctypes.byref(size)):
                raise OSError('Cannot read processor topology')
            return max(1, _count_core_records(buf.raw[:size.value]))
        allowed = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
        cores = set()
        for cpu in sysfs.glob('cpu[0-9]*'):
            if not cpu.name[3:].isdigit() or (allowed is not None and int(cpu.name[3:]) not in allowed):
                continue
            t = cpu / 'topology'
            cores.add(((t / 'physical_package_id').read_text().strip(),
                       (t / 'core_id').read_text().strip()))
        return len(cores) or 1
    except (OSError, ValueError, AttributeError):
        return 1
