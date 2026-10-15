"""Commands. Stage 1 plan 1: status, models, setup."""

from __future__ import annotations

import argparse, pathlib

from workbench import catalog, fetch, hw, packs
from workbench.ui import ask, gb, say

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WORK = ROOT / ".workbench" / "fetch"
CATALOG = ROOT / "catalog.toml"


def _profile_lines(p: hw.Profile) -> None:
    say(f"Machine: {p.os} {p.arch}{' (WSL)' if p.wsl else ''}, {p.cpu_name}, {p.cpu_cores} cores")
    say(f"RAM: {gb(p.ram_bytes)}   free disk for models: {gb(p.disk_free_bytes)}")
    for g in p.gpus:
        vram = gb(g.vram_bytes) if g.vram_bytes else "shared memory"
        say(f"GPU: {g.name} ({vram}{', driver ' + g.driver if g.driver else ''})")
    if not p.gpus:
        say("GPU: none found (models run on the CPU)")
    say(f"Tier: {p.tier}")
    choice = packs.choose_pack(p)
    say(f"Platform pack: {choice.pack or 'not supported'}")
    if choice.note:
        say(f"  {choice.note}")


def plan_downloads(chosen: list[catalog.Model], models_dir: pathlib.Path) -> int:
    return sum(m.total_bytes for m in chosen if not fetch.is_installed(m, models_dir))


def _download(chosen: list[catalog.Model], p: hw.Profile) -> int:
    need = plan_downloads(chosen, MODELS)
    if need * 1.02 > p.disk_free_bytes:
        say(f"Not enough disk: need {gb(need)}, free {gb(p.disk_free_bytes)}. "
            "Free up space or remove models, then run again.")
        return 1
    for m in chosen:
        if fetch.is_installed(m, MODELS):
            say(f"  {m.id}: already installed")
            continue
        say(f"  {m.id}: downloading {gb(m.total_bytes)}")
        try:
            fetch.fetch_model(m, MODELS, WORK, progress=lambda name, done, total: print(
                f"\r    {name}: {done * 100 // total}%", end="", flush=True))
            print()
        except fetch.FetchError as e:
            print()
            say(f"  {m.id}: {e}")
            return 1
    say("Models ready.")
    return 0


def _remove(m: catalog.Model) -> None:
    folder = MODELS / m.folder
    for f in m.files:
        (folder / f.name).unlink(missing_ok=True)
        (folder / f"{f.name}.ok").unlink(missing_ok=True)
    if folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
    say(f"  removed {m.id}")


def cmd_status(a, p, models) -> int:
    _profile_lines(p)
    say("")
    say("Models  ([x] installed, [ ] can run here, [-] cannot run here):")
    for o in catalog.offered(models, p):
        mark = "x" if fetch.is_installed(o.model, MODELS) else (" " if o.runnable else "-")
        extra = f"  ({o.reason})" if o.reason else ""
        say(f"  [{mark}] {o.model.id:<18} {gb(o.model.total_bytes):>8}{extra}")
    return 0


def cmd_models(a, p, models) -> int:
    by_id = {m.id: m for m in models}
    offers = {o.model.id: o for o in catalog.offered(models, p)}
    for mid in (a.add or []) + (a.remove or []):
        if mid not in by_id:
            say(f"Unknown model {mid!r}. Known: {', '.join(by_id)}")
            return 2
    for mid in a.remove or []:
        _remove(by_id[mid])
    if a.add:
        bad = [o for o in (offers[m] for m in a.add) if not o.runnable]
        for o in bad:
            say(f"{o.model.id} cannot run on this machine: {o.reason}")
        if bad:
            return 1
        return _download([by_id[m] for m in a.add], p)
    if a.remove:
        return 0
    rec = catalog.recommended(list(offers.values()), p.tier)
    if a.yes:
        return _download(rec, p)
    say("Models this machine can run (* = recommended):")
    for o in offers.values():
        if o.runnable:
            star = "*" if o.model in rec else " "
            done = " installed" if fetch.is_installed(o.model, MODELS) else ""
            note = f" ({o.reason})" if o.reason else ""
            say(f"  {star} {o.model.id:<18} {gb(o.model.total_bytes):>8}{done}{note}")
    answer = ask("Models to install, comma-separated", ",".join(m.id for m in rec))
    picked = [s.strip() for s in answer.split(",") if s.strip()]
    unknown = [s for s in picked if s not in offers or not offers[s].runnable]
    if unknown:
        say(f"Not available here: {', '.join(unknown)}")
        return 2
    return _download([by_id[s] for s in picked], p)


def cmd_setup(a, p, models) -> int:
    _profile_lines(p)
    choice = packs.choose_pack(p)
    if choice.pack is None:
        return 1
    say("")
    return cmd_models(argparse.Namespace(add=None, remove=None, yes=a.yes), p, models)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="workbench.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("models")
    m.add_argument("--add", nargs="+")
    m.add_argument("--remove", nargs="+")
    m.add_argument("--yes", action="store_true")
    s = sub.add_parser("setup")
    s.add_argument("--yes", action="store_true")
    a = ap.parse_args(argv)
    p = hw.detect(MODELS)
    models = catalog.load(CATALOG)
    return {"status": cmd_status, "models": cmd_models, "setup": cmd_setup}[a.cmd](a, p, models)
