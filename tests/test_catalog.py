import pathlib, tempfile, unittest

from workbench import catalog, hw

FIX = pathlib.Path(__file__).parent / "fixtures"
GB = 1_000_000_000


def profile(ram_gb, vram_gb=0, wsl=False, os="linux"):
    gpus = [hw.Gpu("nvidia", "x", int(vram_gb * GB), "581", 13010, (12, 0), False, "nvml")] if vram_gb else []
    return hw.Profile(os=os, arch="x86_64", wsl=wsl, rosetta=False, cpu_name="x", cpu_cores=8,
                      ram_bytes=int(ram_gb * GB), ram_available_bytes=None, disk_free_bytes=900 * GB,
                      gpus=gpus, vulkan_loader=False)


class LoadTest(unittest.TestCase):
    def test_load_small(self):
        models = catalog.load(FIX / "catalog-small.toml")
        self.assertEqual([m.id for m in models], ["tiny-chat", "mid-chat", "big-moe"])
        tiny = models[0]
        self.assertEqual(tiny.total_bytes, 150)
        self.assertEqual(tiny.files[0].chunks, (catalog.Chunk("b1", 100), catalog.Chunk("b2", 50)))
        self.assertEqual(models[1].files[1].role, "mmproj")

    def test_chunk_sizes_must_add_up(self):
        bad = (FIX / "catalog-small.toml").read_text().replace("bytes = 150", "bytes = 151")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "c.toml"
            p.write_text(bad)
            with self.assertRaisesRegex(catalog.CatalogError, "tiny-chat.*tiny.gguf"):
                catalog.load(p)

    def test_missing_field_names_model(self):
        bad = (FIX / "catalog-small.toml").read_text().replace('engine = "llama.cpp"\nram_gb = 2', "ram_gb = 2", 1)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "c.toml"
            p.write_text(bad)
            with self.assertRaisesRegex(catalog.CatalogError, "tiny-chat.*engine"):
                catalog.load(p)


class OfferTest(unittest.TestCase):
    def setUp(self):
        self.models = catalog.load(FIX / "catalog-small.toml")

    def offers(self, prof):
        return {o.model.id: o for o in catalog.offered(self.models, prof)}

    def test_cpu_laptop(self):
        o = self.offers(profile(ram_gb=16))
        self.assertTrue(o["tiny-chat"].runnable)
        self.assertIsNone(o["tiny-chat"].reason)
        self.assertTrue(o["mid-chat"].runnable)
        self.assertEqual(o["mid-chat"].reason, "runs on CPU only; slow")
        self.assertFalse(o["big-moe"].runnable)
        self.assertIn("needs 40 GB of RAM; this machine has 16", o["big-moe"].reason)

    def test_gpu8_laptop(self):
        o = self.offers(profile(ram_gb=32, vram_gb=8))
        self.assertTrue(o["mid-chat"].runnable)
        self.assertIsNone(o["mid-chat"].reason)
        self.assertFalse(o["big-moe"].runnable)
        self.assertIn("needs 12 GB of video memory; this machine has 8", o["big-moe"].reason)

    def test_recommended_follows_tier(self):
        prof = profile(ram_gb=32, vram_gb=8)
        rec = catalog.recommended(catalog.offered(self.models, prof), prof.tier)
        self.assertEqual([m.id for m in rec], ["tiny-chat", "mid-chat"])

    def test_slow_cpu_fallback_is_not_the_default(self):
        """A model that only fits by spilling to the CPU stays listed as slow."""
        prof = profile(ram_gb=64, vram_gb=8)
        offer = self.offers(prof)["big-moe"]
        self.assertTrue(offer.runnable)
        self.assertIsNotNone(offer.reason)
        self.assertIn("slow", offer.reason or "")
        rec = [m.id for m in catalog.recommended(catalog.offered(self.models, prof), prof.tier)]
        self.assertEqual(rec, ["tiny-chat", "mid-chat"])

    def test_real_catalog_selects_what_fits_the_detected_machine(self):
        """8 GB of video memory and 64 GB of RAM is the other laptop, not a tier tag."""
        models = catalog.load(pathlib.Path(__file__).parents[1] / "catalog.toml")
        gpu = profile(ram_gb=64, vram_gb=8)
        selected = [m.id for m in catalog.recommended(catalog.offered(models, gpu), gpu.tier)]
        for mid in ("gpt-oss-20b", "qwen3.6-35b-a3b", "gemma4-e4b", "granite-4.2-8b"):
            self.assertIn(mid, selected)
        cpu = profile(ram_gb=64)
        offers = {o.model.id: o for o in catalog.offered(models, cpu)}
        self.assertTrue(offers["gpt-oss-20b"].runnable)
        self.assertIsNotNone(offers["gpt-oss-20b"].reason)
        self.assertIn("slow", offers["gpt-oss-20b"].reason or "")
        cpu_selected = [m.id for m in catalog.recommended(catalog.offered(models, cpu), cpu.tier)]
        self.assertNotIn("gpt-oss-20b", cpu_selected)
        self.assertNotIn("qwen3.6-35b-a3b", cpu_selected)

    def test_recommended_never_includes_unrunnable(self):
        prof = profile(ram_gb=4)
        rec = catalog.recommended(catalog.offered(self.models, prof), "gpu-8")
        self.assertEqual([m.id for m in rec], ["tiny-chat"])

    def test_small_gpu_never_offers_less_than_cpu(self):
        """I7: a 2 GB dedicated GPU must still run what the CPU alone could."""
        no_gpu = {k: o.runnable for k, o in self.offers(profile(ram_gb=16)).items()}
        small = self.offers(profile(ram_gb=16, vram_gb=2))
        for mid, ok in no_gpu.items():
            if ok:
                self.assertTrue(small[mid].runnable, mid)
        self.assertEqual(small["mid-chat"].reason, "too big for this GPU; runs on CPU only; slow")

    def test_wsl_reason_mentions_wslconfig(self):
        """Review focus 5: fits the host's RAM but not WSL's half."""
        o = self.offers(profile(ram_gb=16, vram_gb=16, wsl=True))
        self.assertFalse(o["big-moe"].runnable)
        self.assertIn(".wslconfig", o["big-moe"].reason)


if __name__ == "__main__":
    unittest.main()
