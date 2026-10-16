import pathlib, tempfile, unittest, tomllib
from unittest import mock
from workbench import catalog, hw
from workbench.runtime import Config, is_loopback, load_tuning
from tools import catalog_write

class RuntimeTest(unittest.TestCase):
    def test_physical_cores_not_smt_threads(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            for cpu, core in enumerate((0, 0, 1, 1)):
                p = root / f'cpu{cpu}' / 'topology'
                p.mkdir(parents=True)
                (p / 'physical_package_id').write_text('0')
                (p / 'core_id').write_text(str(core))
            with mock.patch('os.sched_getaffinity', return_value={0, 1, 2, 3}, create=True):
                self.assertEqual(hw.physical_cores('linux', root), 2)

    def test_windows_core_records(self):
        from workbench.hw import _count_core_records
        import struct
        data = struct.pack('<II', 0, 16) + bytes(8)
        data += struct.pack('<II', 1, 16) + bytes(8)
        data += struct.pack('<II', 0, 16) + bytes(8)
        self.assertEqual(_count_core_records(data), 2)
        self.assertRaises(ValueError, _count_core_records, b'bad')

    def test_settings_roundtrip_and_defaults(self):
        source = pathlib.Path('tests/fixtures/catalog-small.toml').read_text()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'catalog.toml'
            path.write_text(source.replace('engine = "llama.cpp"',
                'engine = "llama.cpp"\nsettings = { n_ctx = 12288 }', 1))
            self.assertEqual(catalog.load(path)[0].settings, {'n_ctx': 12288})
            self.assertEqual(catalog.load(path)[1].settings, {})
            data = tomllib.loads(path.read_text())
            out = catalog_write.dump(data, '# test\n')
            self.assertEqual(tomllib.loads(out)['model'][0]['settings'], {'n_ctx': 12288})

    def test_bind_validation(self):
        self.assertTrue(is_loopback('::1'))
        self.assertTrue(is_loopback('127.0.0.2'))
        self.assertFalse(is_loopback('localhost.example.org'))
        self.assertRaises(ValueError, Config, pathlib.Path('.'), pathlib.Path('models'), '0.0.0.0')

    def test_tuning_rejects_non_mapping_settings_and_footprint(self):
        import json
        model = catalog.load(pathlib.Path('tests/fixtures/catalog-small.toml'))[0]
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / 'data').mkdir()
            row = {'sha256': {f.name: f.sha256 for f in model.files},
                   'settings': {'n_ctx': 12288}, 'footprint': {'ram': 100, 'vram': 0}}
            for field in ('settings', 'footprint'):
                with self.subTest(field=field):
                    invalid = dict(row, **{field: []})
                    (root / 'data' / 'tuning.json').write_text(
                        json.dumps({model.id: {'cpu': invalid}}))
                    self.assertEqual(load_tuning(root, model, 'cpu'), {})
