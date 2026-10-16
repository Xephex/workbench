"""CI check after `setup --packages-only`: the pack is recorded, its stack imports, no .exe is left,
and Windows has no monty (spec B6, B8). Run from the repo root: python -I tools/check_pack_install.py PACK"""

import importlib.util, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workbench import install  # noqa: E402

pack = sys.argv[1]
assert install.installed_pack(ROOT) == pack, install.installed_pack(ROOT)
assert install.activate(ROOT)
import fastapi, gradio, llama_cpp, opentelemetry.sdk, phoenix  # noqa: E401,F401,E402

print("fastapi", fastapi.__version__, "gradio", gradio.__version__, "llama_cpp", llama_cpp.__version__)
print(llama_cpp.llama_print_system_info().decode())
exes = [str(p) for p in install.site_dir(ROOT).rglob("*") if p.suffix.lower() == ".exe"]
assert not exes, exes
if sys.platform == "win32":
    assert importlib.util.find_spec("pydantic_monty") is None, "monty must not be installed on Windows"
print("ok")
