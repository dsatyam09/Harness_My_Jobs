"""Expose the job-board sandbox as an MCP server (for the copilot and other MCP clients).

    python -m jobharness.mcp_server --data data
"""
from __future__ import annotations

import argparse

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP

from .run import load_board


def build(data_dir: str) -> FastMCP:
    board = load_board(data_dir)
    mcp = FastMCP("jobharness")

    @mcp.tool()
    def search_jobs(query: str, k: int = 10) -> list[dict]:
        """Keyword search over job titles and descriptions."""
        return board.search(query, k)

    @mcp.tool()
    def get_job(job_id: str) -> dict:
        """Read one posting: title, company, location and description."""
        return board.get_job(job_id)

    @mcp.tool()
    def find_evidence(job_id: str, aspect: str) -> dict:
        """Sentences of a posting about salary, remote, location, sponsorship, experience or skills."""
        return board.evidence(job_id, aspect)

    @mcp.tool()
    def get_profile(profile_id: str) -> dict:
        """Read a candidate profile."""
        return board.profile(profile_id)

    return mcp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    build(ap.parse_args().data).run()
