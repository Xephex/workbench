import pathlib, sys, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import catalog_chunks  # noqa: E402

from workbench import catalog  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


class ChunksFromTreeTest(unittest.TestCase):
    TREE = [
        {"path": "m/x.gguf.part001", "type": "blob", "sha": "b", "size": 5},
        {"path": "m/x.gguf.part000", "type": "blob", "sha": "a", "size": 10},
        {"path": "m/x.gguf.sha256", "type": "blob", "sha": "s", "size": 80},
        {"path": "m/y.gguf.part000", "type": "blob", "sha": "c", "size": 1},
    ]

    def test_orders_and_filters(self):
        self.assertEqual(catalog_chunks.chunks_from_tree(self.TREE, "m", "x.gguf"),
                         [{"blob": "a", "bytes": 10}, {"blob": "b", "bytes": 5}])

    def test_gap_raises(self):
        tree = [t for t in self.TREE if not t["path"].endswith("part000")]
        with self.assertRaises(ValueError):
            catalog_chunks.chunks_from_tree(tree, "m", "x.gguf")


class RealCatalogTest(unittest.TestCase):
    def test_real_catalog(self):
        models = {m.id: m for m in catalog.load(ROOT / "catalog.toml")}
        self.assertEqual(len(models), 14)
        main = lambda tier: sum(f.bytes for m in models.values() if tier in m.recommended_for  # noqa: E731
                                for f in m.files if f.role == "main")
        self.assertAlmostEqual(main("cpu") / 1e9, 5.3, delta=0.05)    # spec B3 recommended sets
        self.assertAlmostEqual(main("gpu-8") / 1e9, 9.7, delta=0.05)
        self.assertAlmostEqual(main("gpu-16") / 1e9, 22.8, delta=0.05)
        for mid in ("qwen3-rerank-0.6b", "qwen3-rerank-4b", "qwen3-rerank-8b"):
            self.assertEqual(models[mid].files[0].repo, "Xephex/workbench-models")

    def test_known_files(self):
        models = {m.id: m for m in catalog.load(ROOT / "catalog.toml")}
        self.assertEqual(models["qwen3-embed-0.6b"].files[0].bytes, 639150592)
        self.assertEqual(models["qwen3-embed-0.6b"].files[0].sha256,
                         "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439")
        self.assertEqual(len(models["qwen3-embed-0.6b"].files[0].chunks), 7)
        self.assertEqual(models["qwen3.5-4b"].files[0].bytes, 2740937888)
        self.assertEqual(sum(len(m.files) for m in models.values()), 20)  # 6 models also have a vision file
        new = {"qwen3.5-0.8b", "qwen3.5-2b", "qwen3-embed-4b", "qwen3-embed-8b",
               "qwen3-rerank-0.6b", "qwen3-rerank-4b", "qwen3-rerank-8b"}
        for m in models.values():
            for f in m.files:
                self.assertEqual(f.repo, "Xephex/workbench-models" if m.id in new else "Xephex/playground_stuff")
                self.assertEqual([c.bytes for c in f.chunks[:-1]], [99_614_720] * (len(f.chunks) - 1))


if __name__ == "__main__":
    unittest.main()
