import os, pathlib, runpy
from workbench import install

async def _skip_grpc(self):
    # Phoenix 20.22 binds gRPC to [::] regardless of PHOENIX_HOST.
    # Workbench exports over loopback HTTP, so no gRPC listener is needed.
    return None

def main():
    root = pathlib.Path(__file__).resolve().parents[1]
    if not install.activate(root):
        raise SystemExit('Phoenix requires the installed tracing pack.')
    os.environ['PHOENIX_ALLOW_EXTERNAL_RESOURCES'] = 'false'
    os.environ['PHOENIX_TELEMETRY_ENABLED'] = 'false'
    os.environ['PHOENIX_HOST'] = '127.0.0.1'
    from phoenix.server.grpc_server import GrpcServer
    GrpcServer.__aenter__ = _skip_grpc
    import sys
    sys.argv = ['phoenix', 'serve']
    runpy.run_module('phoenix.server.main', run_name='__main__')

if __name__ == '__main__':
    main()
