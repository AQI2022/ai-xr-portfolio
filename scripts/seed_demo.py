import json
from pathlib import Path

from ai_xr.config import Settings
from ai_xr.rag import chunk_document
from ai_xr.store import Store


def main():
    settings = Settings().prepare()
    store = Store(settings.data_dir / "portfolio.sqlite")
    root = Path(__file__).resolve().parents[1]
    count = 0
    for path in (root / "examples/knowledge").glob("*.md"):
        doc_id, chunks = chunk_document(path.name, path.read_bytes())
        store.replace_document(doc_id, chunks)
        count += len(chunks)
    store.save_jobs(json.loads((root / "examples/jobs.json").read_text(encoding="utf-8")))
    print(json.dumps({"seeded_chunks": count, "sample_jobs": len(store.list_jobs())}))


if __name__ == "__main__":
    main()
