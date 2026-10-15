import io, unittest
from unittest import mock

from workbench import ui


class UiTest(unittest.TestCase):
    def test_gb_formats_one_decimal(self):
        self.assertEqual(ui.gb(5_300_000_000), "5.3 GB")
        self.assertEqual(ui.gb(639_150_592), "0.6 GB")
        self.assertEqual(ui.gb(0), "0.0 GB")

    def test_ask_returns_default_on_empty_input(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(ui.ask("Pick", "y"), "y")

    def test_ask_returns_default_on_eof(self):
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(ui.ask("Pick", "n"), "n")

    def test_say_prints_line(self):
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            ui.say("hello")
        self.assertEqual(buf.getvalue(), "hello\n")


if __name__ == "__main__":
    unittest.main()
