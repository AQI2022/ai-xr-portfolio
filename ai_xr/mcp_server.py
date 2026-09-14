"""Official MCP Python SDK stdio server. Keep stdout exclusively for MCP transport."""
import json

from mcp.server.fastmcp import FastMCP

from .config import Settings
from .jobs import extract_skills, rank_jobs
from .rag import Retriever
from .store import Store

settings = Settings().prepare()
store = Store(settings.data_dir / "portfolio.sqlite")
retriever = Retriever(store, settings.embedding_model, settings.reranker_model)
mcp = FastMCP("AI XR Portfolio")


@mcp.tool()
def search_xr_knowledge(query: str, top_k: int = 4) -> str:
    """Search local XR documents and return source chunks. Read-only."""
    return json.dumps(retriever.search(query[:2000], max(1, min(10, top_k))), ensure_ascii=False)


@mcp.tool()
def analyze_resume_skills(resume: str) -> str:
    """Extract skill names and exact resume evidence without inventing qualifications."""
    return json.dumps(extract_skills(resume[:30000]), ensure_ascii=False)


@mcp.tool()
def match_saved_jobs(resume: str) -> str:
    """Rank already imported job descriptions. Does not submit applications."""
    return json.dumps(rank_jobs(resume[:30000], store.list_jobs()), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
