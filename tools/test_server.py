import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workbench import install
if not install.activate(ROOT):
    raise SystemExit('Install the pack first: python workbench.py setup --packages-only --yes')
if __name__ == '__main__':
    args = sys.argv[1:] or ['discover', '-s', 'tests', '-v']
    unittest.main(module=None, argv=[sys.argv[0], *args])
