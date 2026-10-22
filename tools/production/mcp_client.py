"""已配置本机 Unreal MCP 的串行调度客户端；不自动重试未知结果的写操作。"""
import http.client
import json


def read_sse(stream, request_id):
    """UE先发空HTTP响应头，稍后另写SSE；必须读取底层流而非Content-Length。"""
    data = []
    while True:
        raw = stream.readline(8 * 1024 * 1024)
        if not raw:
            raise EOFError('Unreal MCP在返回请求结果前关闭SSE流')
        line = raw.decode('utf-8').rstrip('\r\n')
        if not line:
            if data:
                message = json.loads('\n'.join(data))
                data = []
                if message.get('id') == request_id:
                    return message
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))


class UnrealMCP:
    def __init__(self, timeout=600):
        self.session = None
        self.version = None
        self.counter = 0
        self.timeout = timeout
        self.descriptions = {}
        initialized = self.request('initialize', {'protocolVersion':'2025-11-25',
            'capabilities':{}, 'clientInfo':{'name':'FutsalMOTProduction','version':'1'}})
        self.version = initialized['protocolVersion']
        self.request('notifications/initialized', {}, notify=True)
        names = {tool['name'] for tool in self.request('tools/list', {})['tools']}
        if not {'call_tool', 'describe_toolset'} <= names:
            raise ValueError('当前MCP缺少已配置的原生工具入口')

    def request(self, method, params, notify=False):
        self.counter += 1
        payload = dict(jsonrpc='2.0', method=method, params=params)
        if not notify:
            payload['id'] = self.counter
        headers = {'Content-Type':'application/json', 'Accept':'application/json, text/event-stream'}
        if self.session:
            headers['Mcp-Session-Id'] = self.session
        if self.version:
            headers['MCP-Protocol-Version'] = self.version
        connection = http.client.HTTPConnection('127.0.0.1', 8000, timeout=self.timeout)
        try:
            connection.request('POST', '/mcp', json.dumps(payload).encode('utf-8'), headers)
            response = connection.getresponse()
            if not 200 <= response.status < 300:
                raise RuntimeError('Unreal MCP HTTP {}'.format(response.status))
            self.session = response.getheader('Mcp-Session-Id', self.session)
            if notify:
                return None
            if 'text/event-stream' in (response.getheader('Content-Type') or ''):
                message = read_sse(response.fp, self.counter)
            else:
                message = json.loads(response.read().decode('utf-8'))
            if message.get('id') != self.counter:
                raise ValueError('MCP响应身份不匹配')
            if 'error' in message:
                raise RuntimeError(message['error'])
            return message['result']
        finally:
            connection.close()

    def describe(self, toolset):
        result = self.request('tools/call', {'name':'describe_toolset', 'arguments':{'toolset_name':toolset}})
        if result.get('isError'):
            raise RuntimeError(result)
        self.descriptions[toolset] = result
        return result

    def call(self, toolset, name, arguments):
        if toolset not in self.descriptions:
            self.describe(toolset)
        result = self.request('tools/call', {'name':'call_tool', 'arguments':{
            'toolset_name':toolset, 'tool_name':name, 'arguments':arguments}})
        if result.get('isError'):
            raise RuntimeError(result)
        value = json.loads(next(item['text'] for item in result['content'] if item['type']=='text'))
        if 'returnValue' not in value:
            raise ValueError('MCP调用返回格式不完整：{}'.format(value))
        return value['returnValue']

    def run_file(self, path):
        result = self.call('futsalmot_tools.FutsalMOTTools', 'run_python_file', {'path':str(path)})
        if result.get('success') != 'true':
            raise RuntimeError(result)
        return result
