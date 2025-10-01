def application(environ, start_response):
    start_response('200 OK', [('Content-Type','application/json')])
    return [b'{"ok": true, "src": "minimal wsgi"}']
