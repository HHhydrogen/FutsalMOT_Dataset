"""覆盖UE原生HTTP先发Content-Length:0、稍后写SSE的实际协议。"""
import importlib.util
import io
from pathlib import Path
import socket
import threading
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest

path = Path(__file__).resolve().parents[1] / 'tools/production/mcp_client.py'
spec = importlib.util.spec_from_file_location('production_mcp', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ignore_progress_and_select_matching_rpc_id():
    data = b'data: {"method":"progress"}\n\ndata: {"id":3,"result":42}\n\n'
    assert module.read_sse(io.BytesIO(data), 3)['result'] == 42


def test_eof_is_failure():
    with pytest.raises(EOFError):
        module.read_sse(io.BytesIO(b''), 1)


def test_actual_delayed_raw_response_after_zero_length_header():
    server = socket.socket()
    server.bind(('127.0.0.1',0))
    server.listen(1)
    ready = threading.Event()
    def send():
        connection, _ = server.accept()
        with connection:
            # 必须消费完整POST；Windows关闭仍有未读请求体的socket会发送RST。
            request = connection.makefile('rb')
            request.readline()
            length = 0
            while True:
                header = request.readline()
                if header == b'\r\n':
                    break
                if header.lower().startswith(b'content-length:'):
                    length = int(header.split(b':',1)[1])
            request.read(length)
            connection.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n')
            ready.wait(3)
            connection.sendall(b'event: message\ndata: {"jsonrpc":"2.0","id":7,"result":{"success":true}}\n\n')
            connection.shutdown(socket.SHUT_WR)
            request.close()
        server.close()
    thread = threading.Thread(target=send)
    thread.start()
    client = http.client.HTTPConnection('127.0.0.1',server.getsockname()[1],timeout=5)
    try:
        client.request('POST','/mcp',b'{}')
        response = client.getresponse()
        assert response.length == 0
        ready.set()
        assert module.read_sse(response.fp,7)['result'] == {'success':True}
    finally:
        ready.set()
        client.close()
        thread.join(5)


def test_full_initialize_describe_and_run_file_over_native_style_sse(monkeypatch):
    received=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):
            pass
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append((payload,dict(self.headers)))
            method=payload['method']
            if method=='notifications/initialized':
                self.send_response(202);self.send_header('Content-Length','0');self.end_headers();return
            if method=='initialize':
                result={'protocolVersion':'2025-11-25'}
            elif method=='tools/list':
                result={'tools':[{'name':'call_tool'},{'name':'describe_toolset'}]}
            elif payload['params']['name']=='describe_toolset':
                result={'content':[{'type':'text','text':'{"tools":["run_python_file"]}'}]}
            else:
                result={'content':[{'type':'text','text':json.dumps({'returnValue':{'success':'true','log':'UE_EXECUTED'}})}]}
            data=json.dumps({'jsonrpc':'2.0','id':payload['id'],'result':result}).encode()
            self.send_response(200)
            self.send_header('Mcp-Session-Id','production-test-session')
            if method=='tools/call':
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Content-Length','0');self.end_headers();self.wfile.flush()
                self.wfile.write(b'event: message\ndata: '+data+b'\n\n');self.wfile.flush()
            else:
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever)
    thread.start()
    original=http.client.HTTPConnection
    def local_connection(host,port,timeout):
        assert (host,port)==('127.0.0.1',8000)
        return original('127.0.0.1',server.server_port,timeout=timeout)
    monkeypatch.setattr(module.http.client,'HTTPConnection',local_connection)
    try:
        client=module.UnrealMCP(timeout=5)
        assert client.run_file('D:/project/control.py')['log']=='UE_EXECUTED'
        assert [item[0]['method'] for item in received]==['initialize','notifications/initialized','tools/list','tools/call','tools/call']
        assert received[-1][0]['params']['arguments']=={'toolset_name':'futsalmot_tools.FutsalMOTTools',
            'tool_name':'run_python_file','arguments':{'path':'D:/project/control.py'}}
        assert received[-1][1]['Mcp-Session-Id']=='production-test-session'
    finally:
        server.shutdown();server.server_close();thread.join(5)
