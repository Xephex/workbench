from __future__ import annotations
import contextlib, json, os, pathlib, subprocess, sys, time, urllib.error, urllib.request, uuid

DEFAULT_BASE = 'http://127.0.0.1:8080'
class ClientError(RuntimeError):
    pass

class Client:
    def __init__(self, base=DEFAULT_BASE, key=None):
        self.base, self.key = base.rstrip('/'), key
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def open(self, method, path, body=None, timeout=3600):
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer '+self.key
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base+path, data=data, headers=headers, method=method)
        try:
            return self.opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            try:
                message = json.loads(exc.read()).get('error', {}).get('message', f'HTTP {exc.code}')
            except ValueError:
                message = f'HTTP {exc.code}'
            raise ClientError(message) from None
    def request(self, method, path, body=None, timeout=3600):
        with self.open(method, path, body, timeout) as response:
            return json.load(response)
    def peek(self):
        try:
            result = self.request('GET', '/wb/status', timeout=1)
        except ClientError as exc:
            # Another program on this port, commonly an older local server, has no Workbench status route.
            if str(exc) == 'HTTP 404':
                return None
            raise
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ConnectionRefusedError):
                return None
            raise ClientError('The server is unreachable or responding too slowly') from None
        except ConnectionRefusedError:
            return None
        except (TimeoutError, ConnectionError):
            raise ClientError('The server is responding too slowly; no second server was started') from None
        if result.get('service') != 'workbench':
            raise ClientError('The selected port is serving another application')
        return result
    def stream(self, path, body, progress=None):
        with self.open('POST', path, dict(body, stream=True), timeout=60) as response:
            data = []
            for raw in response:
                line = raw.decode('utf-8').rstrip('\r\n')
                if line.startswith(':'):
                    if progress:
                        progress(line[1:].strip())
                elif line.startswith('data:'):
                    data.append(line[5:].lstrip())
                elif not line and data:
                    value = '\n'.join(data)
                    data.clear()
                    if value == '[DONE]':
                        return
                    chunk = json.loads(value)
                    if 'error' in chunk:
                        raise ClientError(chunk['error']['message'])
                    yield chunk
            raise ClientError('Server disconnected before [DONE]')

def read_key(root):
    key = os.environ.get('WORKBENCH_API_KEY')
    if key:
        return key
    try:
        return (root/'.workbench/api-key.txt').read_text().strip() or None
    except FileNotFoundError:
        return None

def share_key(root):
    key = read_key(root)
    if key:
        return key
    path = root/'.workbench/api-key.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return read_key(root)
    with os.fdopen(fd, 'w') as file:
        key = uuid.uuid4().hex+uuid.uuid4().hex
        file.write(key+'\n')
    return key

@contextlib.contextmanager
def session(root, model=None, base=DEFAULT_BASE):
    client = Client(base, read_key(root))
    if client.peek() is not None:
        yield client
        return
    if base.rstrip('/') != DEFAULT_BASE:
        raise ClientError('The requested server is not running')
    token = uuid.uuid4().hex
    folder = root/'data/server'
    folder.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(root/'workbench.py'), 'serve', '--host', '127.0.0.1', '--port', '8080']
    if model:
        argv += ['--model', model]
    log = (folder/(token+'.log')).open('ab')
    env = dict(os.environ, WORKBENCH_MANAGED_SERVER='1', WORKBENCH_SERVER_INSTANCE=token)
    try:
        process = subprocess.Popen(argv, cwd=root, env=env, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
    except BaseException:
        log.close()
        raise
    try:
        deadline = time.monotonic()+60
        while time.monotonic() < deadline:
            state = client.peek()
            if state and state['instance'] == token:
                break
            if process.poll() is not None:
                # A concurrent client may have won the bind. Reuse only a
                # verified Workbench server; this process owns no other PID.
                if state:
                    break
                raise ClientError(f'Server failed to start; see {folder/(token+".log")}')
            time.sleep(.1)
        else:
            raise ClientError('Server startup timed out')
        yield client
    finally:
        if process.poll() is None:
            try:
                process.stdin.write('stop\n'); process.stdin.flush()
                process.wait(timeout=30)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        if process.stdin:
            process.stdin.close()
        log.close()
