# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp>=1.2,<2"]
# ///
"""Drive the content-studio MCP server over stdio: python test_client.py tool '{"arg": 1}' ..."""
import asyncio, json, sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).parent

async def main():
    params = StdioServerParameters(command="uv", args=["run", "--script", str(HERE / "server.py")])
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("TOOLS:", [t.name for t in tools.tools])
            calls = sys.argv[1:]
            for i in range(0, len(calls), 2):
                name, args = calls[i], json.loads(calls[i + 1])
                res = await s.call_tool(name, args)
                text = "".join(getattr(c, "text", "") for c in res.content)
                print(f"\n== {name} {'(ERROR)' if res.isError else ''}\n{text[:3000]}")
                try:
                    data = json.loads(text)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("status") == "running":
                    while True:
                        res = await s.call_tool("job_status", {"job_id": data["job_id"], "wait_seconds": 40})
                        text = "".join(getattr(c, "text", "") for c in res.content)
                        d = json.loads(text)
                        print("   ..", d.get("status"), d.get("elapsed_seconds"), d.get("progress"))
                        if d["status"] != "running":
                            print(text[:3000]); break

asyncio.run(main())
