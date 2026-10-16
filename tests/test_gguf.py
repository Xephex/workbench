import io, pathlib, struct, sys, unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import gguf  # noqa: E402


def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _s(key) + struct.pack("<I", vtype) + payload


def _tensor(name: str, dims: list[int]) -> bytes:
    return (_s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<IQ", 0, 0))


def gguf_bytes(pool: int = 4, cls: bool = True, arch: str = "qwen3") -> bytes:
    """GGUF v3 header bytes: a few kv types (string, uint32, float32, bool, arrays) and two or three tensors."""
    kvs = [
        _kv("general.architecture", 8, _s(arch)),
        _kv(f"{arch}.pooling_type", 4, struct.pack("<I", pool)),
        _kv(f"{arch}.rope.freq_base", 6, struct.pack("<f", 1e6)),
        _kv("tokenizer.ggml.add_eos_token", 7, b"\x01"),
        _kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + _s("a") + _s("b") + _s("<s>")),
        _kv("tokenizer.ggml.token_type", 9, struct.pack("<IQ", 5, 3) + struct.pack("<3i", 1, 1, 3)),
        _kv("general.file_type", 10, struct.pack("<Q", 15)),
    ]
    tensors = [_tensor("token_embd.weight", [8, 3]), _tensor("blk.0.attn_q.weight", [8, 8])]
    if cls:
        tensors.append(_tensor("cls.output.weight", [8, 1]))
    return b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(tensors)


def buf(**kw) -> io.BytesIO:
    return io.BytesIO(gguf_bytes(**kw) + b"\0" * 64)  # tensor data would follow the header


class GgufTest(unittest.TestCase):
    def test_reads_kv_and_tensor_names(self):
        kv, tensors = gguf.read_header(buf())
        self.assertEqual(kv["general.architecture"], "qwen3")
        self.assertEqual(kv["qwen3.pooling_type"], 4)
        self.assertEqual(kv["tokenizer.ggml.tokens"], ["a", "b", "<s>"])
        self.assertEqual(kv["tokenizer.ggml.token_type"], [1, 1, 3])
        self.assertIs(kv["tokenizer.ggml.add_eos_token"], True)
        self.assertEqual(kv["general.file_type"], 15)
        self.assertEqual(tensors, ["token_embd.weight", "blk.0.attn_q.weight", "cls.output.weight"])

    def test_reranker_with_head_passes(self):
        self.assertIsNone(gguf.rerank_problem(*gguf.read_header(buf(pool=4, cls=True))))

    def test_reranker_without_head_refused(self):
        self.assertIn("cls.output.weight", gguf.rerank_problem(*gguf.read_header(buf(pool=4, cls=False))))

    def test_wrong_pooling_refused(self):
        self.assertIn("pooling", gguf.rerank_problem(*gguf.read_header(buf(pool=3, cls=True))))

    def test_not_gguf(self):
        self.assertRaises(ValueError, gguf.read_header, io.BytesIO(b"PK\x03\x04"))

    def test_truncated_header(self):
        self.assertRaises(ValueError, gguf.read_header, io.BytesIO(gguf_bytes()[:-10]))


if __name__ == "__main__":
    unittest.main()
