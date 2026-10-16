from __future__ import annotations
import dataclasses, json, os, pathlib, sys, threading, uuid
from workbench import catalog, fetch, install
from workbench.client import Client, ClientError, DEFAULT_BASE, read_key, share_key, session
from workbench.runtime import Config

def add_parsers(sub):
    serve = sub.add_parser('serve')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8080)
    serve.add_argument('--share', action='store_true')
    serve.add_argument('--no-trace-content', action='store_true')
    serve.add_argument('--no-trace-viewer', action='store_true')
    parsers = [serve]
    for name in ('chat', 'ask', 'embed', 'rank'):
        parser = sub.add_parser(name)
        parser.add_argument('--server', default=DEFAULT_BASE)
        parsers.append(parser)
        if name == 'chat':
            parser.add_argument('--system', default='You are a helpful assistant.')
        if name == 'ask':
            parser.add_argument('text')
        if name == 'embed':
            parser.add_argument('texts', nargs='+')
        if name == 'rank':
            parser.add_argument('query')
            parser.add_argument('documents', nargs='+')
            parser.add_argument('--top-n', type=int)
        if name in ('chat', 'ask'):
            parser.add_argument('--think', action='store_true', default=None)
    for parser in parsers:
        parser.add_argument('--model')
        parser.add_argument('--json', action='store_true')

def serve(a, root, models_dir):
    if not install.activate(root):
        raise ClientError('Packages are not installed. Run setup first.')
    # Third-party imports occur only after activation, including on Windows.
    from workbench.server.app import create_app
    import uvicorn
    key = share_key(root) if a.share else read_key(root)
    config = Config(root, models_dir, host='0.0.0.0' if a.share else a.host,
        port=a.port, key=key, capture=not a.no_trace_content, phoenix=not a.no_trace_viewer,
        default_model=a.model)
    app = create_app(config)
    app.state.instance = os.environ.get('WORKBENCH_SERVER_INSTANCE', app.state.instance)
    server = uvicorn.Server(uvicorn.Config(app, host=config.host, port=config.port,
        log_level='warning', access_log=False))
    if os.environ.get('WORKBENCH_MANAGED_SERVER') == '1':
        def watch_owner():
            sys.stdin.readline()
            server.should_exit = True
        threading.Thread(target=watch_owner, daemon=True).start()
    host = f'[{config.host}]' if ':' in config.host else config.host
    address = f'http://{host}:{config.port}'
    if a.json:
        print(json.dumps({'address': address, 'authenticated': bool(key)}), flush=True)
    else:
        print(f'Serving at {address}; Ctrl+C stops the server.', flush=True)
        if a.share:
            print('Sharing requires the key in WORKBENCH_API_KEY or .workbench/api-key.txt.', flush=True)
    try:
        server.run()
    finally:
        import asyncio
        asyncio.run(app.state.cleanup())
    return 0

def answer(client, a, messages):
    body = {'messages': messages}
    if a.model:
        body['model'] = a.model
    if a.think is not None:
        body['think'] = a.think
    if a.json:
        result = client.request('POST', '/v1/chat/completions', body)
        print(json.dumps(result, ensure_ascii=False))
        return result['choices'][0]['message']
    chunks = []
    last_phase = None
    def progress(phase):
        nonlocal last_phase
        if phase != last_phase:
            label = {'prompt': 'Reading the prompt', 'queued': 'Waiting for model',
                     'loading': 'Loading model', 'generation': 'Generating'}
            print(label.get(phase, phase), file=sys.stderr, flush=True)
            last_phase = phase
    for chunk in client.stream('/v1/chat/completions', body, progress):
        for choice in chunk.get('choices', []):
            piece = choice['delta'].get('content') or ''
            chunks.append(piece)
            print(piece, end='', flush=True)
            for call in choice['delta'].get('tool_calls') or []:
                print(json.dumps({'tool_call': call}), flush=True)
    print()
    return {'role': 'assistant', 'content': ''.join(chunks)}

def dispatch(a, root, models_dir):
    try:
        if a.cmd == 'serve':
            return serve(a, root, models_dir)
        default_model = a.model if a.cmd in ('ask', 'chat') else None
        with session(root, default_model, a.server) as client:
            if a.cmd == 'ask':
                answer(client, a, [{'role': 'user', 'content': a.text}])
            elif a.cmd == 'chat':
                messages = [{'role': 'system', 'content': a.system}]
                try:
                    while True:
                        try:
                            prompt = input('you> ' if not a.json else '').strip()
                        except EOFError:
                            break
                        if prompt in ('exit', 'quit'):
                            break
                        if prompt == '/reset':
                            messages = messages[:1]
                        elif prompt:
                            messages.append({'role': 'user', 'content': prompt})
                            messages.append(answer(client, a, messages))
                finally:
                    folder = root/'data/chats'
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder/(uuid.uuid4().hex+'.json')).write_text(json.dumps(
                        {'model': a.model, 'messages': messages}, ensure_ascii=False, indent=2), encoding='utf-8')
            elif a.cmd == 'embed':
                body = {'input': a.texts}
                if a.model:
                    body['model'] = a.model
                result = client.request('POST', '/v1/embeddings', body)
                if a.json:
                    print(json.dumps(result))
                else:
                    for row in result['data']:
                        print(json.dumps(row['embedding']))
            elif a.cmd == 'rank':
                body = {'query': a.query, 'documents': a.documents}
                if a.model:
                    body['model'] = a.model
                if a.top_n is not None:
                    body['top_n'] = a.top_n
                result = client.request('POST', '/v1/rerank', body)
                if a.json:
                    print(json.dumps(result))
                else:
                    for row in result['results']:
                        print(f"{row['index']}\t{row['relevance_score']:.6f}\t{a.documents[row['index']]}")
        return 0
    except (ClientError, OSError, ValueError) as exc:
        if a.json:
            print(json.dumps({'error': str(exc)}))
        else:
            print(str(exc), file=sys.stderr)
        return 1

def status(a, profile, models, root, models_dir):
    from workbench.cli import cmd_status
    server = Client(a.server, read_key(root)).peek()
    chosen = [m for m in models if not a.model or m.id == a.model]
    if a.json:
        print(json.dumps({'profile': dataclasses.asdict(profile), 'server': server,
            'models': [{'id': m.id, 'installed': fetch.is_installed(m, models_dir)} for m in chosen]}))
    else:
        cmd_status(a, profile, chosen)
        print('Server: '+('not running' if server is None else json.dumps(server)))
    return 0
