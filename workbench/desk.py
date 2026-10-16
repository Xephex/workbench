"""Mouse-capable menu. Standard library only, so it works before setup installs packages.

Windows Terminal and other virtual terminals report clicks as SGR mouse sequences
once virtual-terminal input is enabled. Tk is not required.
"""

from __future__ import annotations

import dataclasses, os, re, sys

from workbench import catalog, fetch, hw
from workbench.ui import gb

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
WIDTH = 78


@dataclasses.dataclass(frozen=True)
class Item:
    id: str
    label: str
    detail: str = ""
    check: bool | None = None
    enabled: bool = True


@dataclasses.dataclass(frozen=True)
class Frame:
    title: str
    info: tuple[str, ...]
    items: tuple[Item, ...]
    cursor: int
    footer: str


@dataclasses.dataclass(frozen=True)
class Up:
    pass


@dataclasses.dataclass(frozen=True)
class Down:
    pass


@dataclasses.dataclass(frozen=True)
class Activate:
    pass


@dataclasses.dataclass(frozen=True)
class Toggle:
    pass


@dataclasses.dataclass(frozen=True)
class Quit:
    pass


@dataclasses.dataclass(frozen=True)
class Click:
    x: int
    y: int


Event = Up | Down | Activate | Toggle | Quit | Click


@dataclasses.dataclass(frozen=True)
class Action:
    name: str
    ids: tuple[str, ...] = ()


def visible(text: str) -> str:
    return _ANSI.sub("", text)


def parse_events(buf: bytearray) -> list[Event]:
    """Take complete key and mouse sequences out of buf. Leave a partial sequence in place."""
    events: list[Event] = []
    while buf:
        if buf[0] != 0x1B:
            byte = buf.pop(0)
            if byte in (3, ord("q"), ord("Q")):
                events.append(Quit())
            elif byte == ord(" "):
                events.append(Toggle())
            elif byte in (ord("\r"), ord("\n")):
                events.append(Activate())
            elif byte in (ord("k"), ord("K")):
                events.append(Up())
            elif byte in (ord("j"), ord("J")):
                events.append(Down())
            continue
        if len(buf) < 2:
            break
        if buf[1] == ord("O") and len(buf) >= 3 and buf[2] in (ord("A"), ord("B")):
            events.append(Up() if buf[2] == ord("A") else Down())
            del buf[:3]
            continue
        if buf[1] != ord("["):
            del buf[:1]
            continue
        if len(buf) >= 3 and buf[2] in (ord("A"), ord("B")):
            events.append(Up() if buf[2] == ord("A") else Down())
            del buf[:3]
            continue
        if len(buf) < 3 or buf[2] != ord("<"):
            if len(buf) < 3:
                break
            del buf[:1]
            continue
        end = next((i for i, b in enumerate(buf) if b in (ord("M"), ord("m"))), None)
        if end is None:
            break
        piece = bytes(buf[3:end]).decode("ascii", "replace")
        final = buf[end]
        del buf[:end + 1]
        parts = piece.split(";")
        if final != ord("M") or len(parts) != 3:
            continue
        try:
            button, x, y = (int(part) for part in parts)
        except ValueError:
            continue
        if button & 32 or button & 3:
            continue
        events.append(Click(x, y))
    return events


def _clip(text: str, width: int = WIDTH) -> str:
    plain = visible(text)
    if len(plain) <= width:
        return text
    return plain[:width]


def paint(frame: Frame, width: int = WIDTH) -> tuple[list[str], dict[int, int]]:
    lines = ["", f"  {frame.title}", "  " + "-" * min(width - 2, 40), ""]
    lines.extend(_clip("  " + line, width) for line in frame.info)
    lines.append("")
    hit: dict[int, int] = {}
    for index, item in enumerate(frame.items):
        hit[len(lines)] = index
        if item.check is None:
            box = "    " if item.enabled else "[-] "
        else:
            box = "[x] " if item.check else "[ ] "
        mark = "> " if index == frame.cursor else "  "
        detail = f"  {item.detail}" if item.detail else ""
        row = f"  {mark}{box}{item.label}{detail}"
        if index == frame.cursor and item.enabled:
            row = "\x1b[7m" + row + "\x1b[0m"
        elif not item.enabled:
            row = "\x1b[2m" + row + "\x1b[0m"
        lines.append(_clip(row, width))
    lines.extend(["", "  " + frame.footer])
    return lines, hit


def _move(frame: Frame, step: int) -> Frame:
    if not frame.items:
        return frame
    index = frame.cursor
    for _ in frame.items:
        index = (index + step) % len(frame.items)
        if frame.items[index].enabled:
            return dataclasses.replace(frame, cursor=index)
    return frame


def _toggle(frame: Frame) -> Frame:
    item = frame.items[frame.cursor]
    if item.check is None or not item.enabled:
        return frame
    items = list(frame.items)
    items[frame.cursor] = dataclasses.replace(item, check=not item.check)
    return dataclasses.replace(frame, items=tuple(items))


def _checked(frame: Frame) -> tuple[str, ...]:
    return tuple(item.id for item in frame.items if item.check)


def apply(frame: Frame, event: Event, hit: dict[int, int] | None = None) -> tuple[Frame, Action | None]:
    if isinstance(event, Quit):
        return frame, Action("quit")
    if isinstance(event, Up):
        return _move(frame, -1), None
    if isinstance(event, Down):
        return _move(frame, 1), None
    if isinstance(event, Click):
        if hit is None:
            return frame, None
        index = hit.get(event.y - 1)
        if index is None:
            return frame, None
        frame = dataclasses.replace(frame, cursor=index)
        item = frame.items[index]
        if not item.enabled:
            return frame, None
        if item.check is not None:
            return _toggle(frame), None
        event = Activate()
    if isinstance(event, Toggle):
        if frame.items[frame.cursor].check is not None:
            return _toggle(frame), None
        event = Activate()
    if isinstance(event, Activate):
        item = frame.items[frame.cursor]
        if not item.enabled:
            return frame, None
        if item.check is not None:
            return _toggle(frame), None
        ids = _checked(frame) if item.id == "go" else ()
        return frame, Action(item.id, ids)
    return frame, None


def home_frame(info: list[str]) -> Frame:
    items = (
        Item("setup", "Setup", "Install packages and choose models"),
        Item("update", "Update", "Pull the latest workbench"),
        Item("models", "Models", "Choose what to download"),
        Item("status", "Status", "This machine and installed models"),
        Item("serve", "Serve", "Local server on 127.0.0.1:8080"),
        Item("chat", "Chat", "Talk through the local server"),
        Item("ask", "Ask", "One question through the local server"),
        Item("quit", "Quit", "Leave"),
    )
    return Frame("Workbench", tuple(info), items, 0,
                 "Click a row, or use the arrow keys. Enter selects. q quits.")


def models_frame(offers: list[catalog.Offer], info: list[str], installed: set[str], *, setup: bool) -> Frame:
    picked = {m.id for m in catalog.recommended(offers, "")}
    items: list[Item] = []
    for offer in offers:
        model = offer.model
        size = gb(model.total_bytes)
        if not offer.runnable:
            items.append(Item(model.id, model.id, f"{size}  {offer.reason}", enabled=False))
            continue
        notes = []
        if model.id in installed:
            notes.append("installed")
        notes.append(offer.reason or "fits this machine")
        items.append(Item(
            model.id, model.id, f"{size}  " + "  ".join(notes),
            check=model.id in picked and model.id not in installed,
        ))
    if setup:
        items.append(Item("go", "Install packages and download the checked models"))
        items.append(Item("packages", "Install packages only"))
    else:
        items.append(Item("go", "Download the checked models"))
    items.append(Item("back", "Back"))
    first = next(i for i, item in enumerate(items) if item.enabled)
    title = "Setup" if setup else "Models"
    return Frame(title, tuple(info), tuple(items), first,
                 "Click a checkbox to change it. The highlighted button runs that action.")


def _enable_windows_vt():
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    standard_input, standard_output = -10, -11
    handle_in = kernel.GetStdHandle(standard_input)
    handle_out = kernel.GetStdHandle(standard_output)
    mode_in, mode_out = wintypes.DWORD(), wintypes.DWORD()
    if not kernel.GetConsoleMode(handle_in, ctypes.byref(mode_in)):
        return None
    kernel.GetConsoleMode(handle_out, ctypes.byref(mode_out))
    # Mouse input as console records would hide the SGR clicks. Virtual-terminal input reports them as text.
    kernel.SetConsoleMode(handle_in, (mode_in.value | 0x0200) & ~0x0010)
    kernel.SetConsoleMode(handle_out, mode_out.value | 0x0004)
    return handle_in, mode_in.value, handle_out, mode_out.value


class Screen:
    def __init__(self, stdin, stdout):
        self.stdin = stdin
        self.stdout = stdout
        self._buf = bytearray()
        self._pending: list[Event] = []
        self._old = None
        self._windows = None

    def __enter__(self):
        if os.name == "nt":
            self._windows = _enable_windows_vt()
        if self.stdin.isatty() and os.name != "nt":
            import termios, tty
            self._old = termios.tcgetattr(self.stdin.fileno())
            tty.setraw(self.stdin.fileno())
        self.stdout.write("\x1b[?1049h\x1b[?25l\x1b[?1000h\x1b[?1006h")
        self.stdout.flush()
        return self

    def __exit__(self, *exc):
        self.stdout.write("\x1b[?1000l\x1b[?1006l\x1b[?25h\x1b[?1049l")
        self.stdout.flush()
        if self._old is not None:
            import termios
            termios.tcsetattr(self.stdin.fileno(), termios.TCSADRAIN, self._old)
        if self._windows is not None:
            import ctypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            handle_in, mode_in, handle_out, mode_out = self._windows
            kernel.SetConsoleMode(handle_in, mode_in)
            kernel.SetConsoleMode(handle_out, mode_out)

    def draw(self, frame: Frame) -> dict[int, int]:
        lines, hit = paint(frame)
        self.stdout.write("\x1b[H" + "\r\n".join(lines) + "\x1b[J")
        self.stdout.flush()
        return hit

    def read(self) -> Event:
        while True:
            if self._pending:
                return self._pending.pop(0)
            self._pending.extend(parse_events(self._buf))
            if self._pending:
                return self._pending.pop(0)
            piece = self.stdin.buffer.read(1)
            if not piece:
                return Quit()
            self._buf.extend(piece)


def session(screen: Screen, frame: Frame) -> tuple[Action, Frame]:
    while True:
        hit = screen.draw(frame)
        frame, action = apply(frame, screen.read(), hit)
        if action is not None:
            return action, frame


def run(root, models_dir, catalog_path) -> int:
    from workbench import cli
    from workbench.tools import commands
    from workbench.update import git_update
    from workbench.ui import say

    while True:
        profile = hw.detect(models_dir)
        models = catalog.load(catalog_path)
        info = cli.machine_lines(profile)
        with Screen(sys.stdin, sys.stdout) as screen:
            action, _frame = session(screen, home_frame(info))
            while action.name in ("setup", "models", "back"):
                if action.name == "back":
                    action, _frame = session(screen, home_frame(info))
                    continue
                offers = catalog.offered(models, profile)
                installed = {m.id for m in models if fetch.is_installed(m, models_dir)}
                choice, picked = session(screen, models_frame(offers, info, installed, setup=action.name == "setup"))
                if choice.name == "back":
                    action, _frame = session(screen, home_frame(info))
                    continue
                action = Action(choice.name if choice.name != "go" else action.name, choice.ids)
                break
        if action.name == "quit":
            return 0
        if action.name == "update":
            code, text = git_update(root)
            say(text)
            if code == 0 and not text.startswith("Already up to date"):
                say("Updated. Start workbench.py again to use the new version.")
                return 0
            _pause()
            continue
        if action.name == "status":
            cli.cmd_status(None, profile, models)
            _pause()
            continue
        if action.name in ("setup", "packages"):
            code = _setup(root, profile, models, action.ids, packages_only=action.name == "packages")
            say("Setup finished." if code == 0 else "Setup did not finish.")
            _pause()
            continue
        if action.name == "models":
            code = cli._download([m for m in models if m.id in action.ids], profile)
            if code == 0 and not action.ids:
                say("Nothing was checked, so nothing was downloaded.")
            _pause()
            continue
        if action.name == "serve":
            say("Starting the server. Ctrl+C returns to the menu.")
            try:
                commands.serve(_namespace(cmd="serve"), root, models_dir)
            except (commands.ClientError, KeyboardInterrupt) as exc:
                if not isinstance(exc, KeyboardInterrupt):
                    say(str(exc))
            _pause()
            continue
        if action.name == "chat":
            try:
                commands.dispatch(_namespace(cmd="chat", system="You are a helpful assistant."), root, models_dir)
            except (commands.ClientError, KeyboardInterrupt) as exc:
                if not isinstance(exc, KeyboardInterrupt):
                    say(str(exc))
            _pause()
            continue
        if action.name == "ask":
            try:
                text = input("Question: ")
            except (EOFError, KeyboardInterrupt):
                continue
            if text.strip():
                try:
                    commands.dispatch(_namespace(cmd="ask", text=text.strip()), root, models_dir)
                except (commands.ClientError, KeyboardInterrupt) as exc:
                    if not isinstance(exc, KeyboardInterrupt):
                        say(str(exc))
            _pause()
            continue
        say(f"Unknown menu action {action.name}.")
        return 2


def _namespace(**kwargs):
    import argparse
    base = dict(host="127.0.0.1", port=8080, share=False, no_trace_content=False, no_trace_viewer=False,
                server="http://127.0.0.1:8080", model=None, json=False, think=None, system="", text="")
    base.update(kwargs)
    return argparse.Namespace(**base)


def _pause() -> None:
    try:
        input("Press Enter to return to the menu. ")
    except (EOFError, KeyboardInterrupt):
        print()


def _setup(root, profile: hw.Profile, models, ids: tuple[str, ...], *, packages_only: bool) -> int:
    from workbench import cli, install, packs
    from workbench.tools import commands
    from workbench.ui import say
    choice = packs.choose_pack(profile)
    if choice.pack is None:
        say(choice.note or "This machine is not supported.")
        return 1
    try:
        if commands.Client(key=commands.read_key(root)).peek() is not None:
            say("Stop the running server before setup changes its installed packages.")
            return 1
        if commands.port_open():
            say("127.0.0.1:8080 is open but is not answering as Workbench. Setup will continue. "
                "Stop that program before serve.")
    except (commands.ClientError, OSError, ValueError) as exc:
        say(str(exc))
        return 1
    for line in cli.machine_lines(profile):
        say(line)
    say("")
    say(f"Installing the {choice.pack} packages (offline, from this folder)...")
    try:
        state = install.install_pack(root, choice.pack, with_tracing=True)
    except install.InstallError as exc:
        say(f"  {exc}")
        return 1
    say(f"  Packages {state}.")
    if packages_only:
        return 0
    if not ids:
        say("No models checked, so none were downloaded.")
        return 0
    by_id = {model.id: model for model in models}
    return cli._download([by_id[mid] for mid in ids if mid in by_id], profile)
