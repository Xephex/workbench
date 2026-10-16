import pathlib, sys, tempfile, tomllib, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import catalog_write  # noqa: E402

from workbench import catalog  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

NEW = {
    "id": "new-model", "kinds": ["rerank"], "engine": "llama.cpp", "ram_gb": 2, "vram_gb": 1.5,
    "recommended_for": [], "license": "Apache-2.0", "source": "example/new@" + "a" * 40,
    "files": [{"name": "new.gguf", "role": "main", "sha256": "b" * 64, "bytes": 15, "repo": "Xephex/workbench-models",
               "chunks": [{"blob": "c" * 40, "bytes": 10}, {"blob": "d" * 40, "bytes": 5}]}],
}


class CatalogWriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "catalog.toml"
        self.text = (ROOT / "catalog.toml").read_text(encoding="utf-8")
        self.path.write_text(self.text, encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_is_byte_identical(self):
        text = (ROOT / "catalog.toml").read_text(encoding="utf-8")
        self.assertEqual(catalog_write.dump(tomllib.loads(text), text.splitlines(keepends=True)[0]), text)

    def test_upsert_replaces_and_appends(self):
        before = [m.id for m in catalog.load(self.path)]
        changed = dict(tomllib.loads(self.text)["model"][0], ram_gb=7.5)
        catalog_write.upsert(self.path, changed)
        catalog_write.upsert(self.path, NEW)
        models = catalog.load(self.path)
        self.assertEqual([m.id for m in models], before + ["new-model"])
        self.assertEqual(models[0].ram_gb, 7.5)
        self.assertEqual(models[-1].files[0].chunks, (catalog.Chunk("c" * 40, 10), catalog.Chunk("d" * 40, 5)))
        self.assertEqual(models[-1].vram_gb, 1.5)
        # only the replaced model's ram_gb line changed; the new model was appended at the end
        text = self.path.read_text(encoding="utf-8")
        self.assertTrue(text.startswith(self.text.splitlines(keepends=True)[0]))
        self.assertEqual(text.count("[[model]]"), self.text.count("[[model]]") + 1)
        catalog_write.upsert(self.path, NEW)  # same entry again: no change
        self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_upsert_refuses_hand_edits_it_would_lose(self):
        self.path.write_text(self.text.replace("[[model]]\n", "[[model]]\n# a hand-written note\n", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hand"):
            catalog_write.upsert(self.path, NEW)


if __name__ == "__main__":
    unittest.main()
