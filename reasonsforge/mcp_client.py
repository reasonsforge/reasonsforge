"""MCP client bridge for connecting to MCP servers as data source adapters.

Wraps the async MCP SDK with synchronous helpers using a background
event loop thread, so ask.py can dispatch tool calls to any MCP server.

Requires: pip install 'mcp>=1.0'
"""

import asyncio
import shlex
import threading

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _mcp_available = True
except ImportError:
    _mcp_available = False


def _require_mcp():
    if not _mcp_available:
        raise ImportError(
            "mcp is required for MCP server support. "
            "Install it with: pip install 'mcp>=1.0'"
        )


class McpBridge:
    """Synchronous bridge to an MCP server over stdio.

    Runs a persistent asyncio event loop in a background thread
    to keep the MCP session alive across multiple tool calls.
    """

    def __init__(self, command):
        _require_mcp()
        self.command = command
        self._tools = []
        self._instructions = ""
        self._loop = None
        self._thread = None
        self._session = None
        self._shutdown = None
        self._ready = threading.Event()
        self._error = None

    def connect(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=30):
            raise TimeoutError(f"MCP server '{self.command}' did not respond within 30s")
        if self._error:
            raise self._error

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._session_lifecycle())

    async def _session_lifecycle(self):
        parts = shlex.split(self.command)
        server_params = StdioServerParameters(
            command=parts[0], args=parts[1:],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self._session = session
                try:
                    result = await session.initialize()
                    if hasattr(result, "instructions") and result.instructions:
                        self._instructions = result.instructions

                    tools_result = await session.list_tools()
                    self._tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "input_schema": t.inputSchema,
                        }
                        for t in tools_result.tools
                    ]

                    self._shutdown = asyncio.Event()
                    self._ready.set()
                    await self._shutdown.wait()
                except Exception as e:
                    self._error = e
                    self._ready.set()

    def list_tools(self):
        return self._tools

    def get_instructions(self):
        return self._instructions

    def call_tool(self, name, arguments):
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(name, arguments), self._loop
        )
        return future.result(timeout=60)

    async def _call_tool_async(self, name, arguments):
        result = await self._session.call_tool(name, arguments)
        parts = []
        for content in result.content:
            if hasattr(content, "text"):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return "\n".join(parts)

    def close(self):
        if self._loop and self._shutdown:
            self._loop.call_soon_threadsafe(self._shutdown.set)
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._loop:
            self._loop.close()
            self._loop = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()
