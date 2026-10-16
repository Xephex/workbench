import pathlib, unittest

from workbench import catalog, desk, hw

FIX = pathlib.Path(__file__).parent / "fixtures"
GB = 1_000_000_000


def profile(ram_gb, vram_gb=0):
    gpus = [hw.Gpu("nvidia", "RTX PRO 2000 Blackwell", int(vram_gb * GB), "570", 12080, (12, 0), False, "nvml")] if vram_gb else []
    return hw.Profile(os="windows" if vram_gb else "linux", arch="x86_64", wsl=False, rosetta=False,
                      cpu_name="x", cpu_cores=8, ram_bytes=int(ram_gb * GB), ram_available_bytes=None,
                      disk_free_bytes=900 * GB, gpus=gpus, vulkan_loader=False)


class ParseTest(unittest.TestCase):
    def test_mouse_click_and_arrows(self):
        buf = bytearray(b"\x1b[<0;4;8M\x1b[A\x1bOB j")
        events = desk.parse_events(buf)
        self.assertEqual(events, [desk.Click(4, 8), desk.Up(), desk.Down(), desk.Toggle(), desk.Down()])
        self.assertEqual(buf, bytearray())

    def test_partial_mouse_sequence_waits(self):
        buf = bytearray(b"\x1b[<0;4")
        self.assertEqual(desk.parse_events(buf), [])
        self.assertTrue(bytes(buf).startswith(b"\x1b[<0;4"))

    def test_mouse_release_and_drag_do_not_click(self):
        buf = bytearray(b"\x1b[<0;4;8m\x1b[<32;4;8M")
        self.assertEqual(desk.parse_events(buf), [])


class MenuTest(unittest.TestCase):
    def setUp(self):
        self.models = catalog.load(FIX / "catalog-small.toml")

    def _frame(self, ram, vram=0):
        prof = profile(ram, vram)
        offers = catalog.offered(self.models, prof)
        return desk.models_frame(offers, ["machine"], installed=set(), setup=True), offers

    def test_gpu_fit_starts_checked_and_slow_does_not(self):
        frame, _offers = self._frame(64, 8)
        by_id = {item.id: item for item in frame.items}
        self.assertTrue(by_id["tiny-chat"].check)
        self.assertTrue(by_id["mid-chat"].check)
        self.assertFalse(by_id["big-moe"].check)
        self.assertTrue(by_id["big-moe"].enabled)
        self.assertIn("slow", by_id["big-moe"].detail)

    def test_click_toggles_a_checkbox_and_activates_a_button(self):
        frame, _offers = self._frame(64, 8)
        lines, hit = desk.paint(frame)
        model_line = next(line for line, index in hit.items() if frame.items[index].id == "mid-chat")
        button_line = next(line for line, index in hit.items() if frame.items[index].id == "go")
        frame, action = desk.apply(frame, desk.Click(2, model_line + 1), hit)
        self.assertIsNone(action)
        self.assertFalse(next(item.check for item in frame.items if item.id == "mid-chat"))
        _frame, action = desk.apply(frame, desk.Click(2, button_line + 1), hit)
        self.assertEqual(action.name, "go")
        self.assertIn("tiny-chat", action.ids)
        self.assertNotIn("mid-chat", action.ids)
        self.assertNotIn("big-moe", action.ids)
        self.assertIn("Install packages", visible_join(lines))

    def test_unrunnable_model_is_shown_and_cannot_be_checked(self):
        frame, _offers = self._frame(4)
        by_id = {item.id: item for item in frame.items}
        self.assertFalse(by_id["big-moe"].enabled)
        self.assertIn("needs", by_id["big-moe"].detail)
        lines, hit = desk.paint(frame)
        line = next(n for n, index in hit.items() if frame.items[index].id == "big-moe")
        frame, action = desk.apply(frame, desk.Click(2, line + 1), hit)
        self.assertIsNone(action)
        self.assertFalse(frame.items[hit[line]].enabled)


def visible_join(lines):
    return "\n".join(desk.visible(line) for line in lines)


if __name__ == "__main__":
    unittest.main()
