import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

from ai_xr.rag import Retriever, chunk_document
from ai_xr.store import Store

CASES = [
    ("HoloLens 跟踪丢失如何处理", "spatial_anchors.md"),
    ("localPosition 与 position 坐标有什么区别", "spatial_anchors.md"),
    ("Profiler 如何分析 Unity 卡顿", "performance.md"),
    ("Unity 网络请求如何避免主线程阻塞", "performance.md"),
    ("RAG 的 Recall 和 MRR 是什么", "rag_evaluation.md"),
    ("RRF hybrid search 和 reranker", "rag_evaluation.md"),
    ("制氢设备培训演示有哪些步骤", "assembly.md"),
    ("场景动作确认令牌如何过期", "assembly.md"),
    ("STT TTS 语音数字人延迟", "voice.md"),
    ("声音克隆和系统 TTS 的区别", "voice.md"),
    ("LoRA 低秩矩阵和 QLoRA 量化", "lora.md"),
    ("微调 prompt token 为什么设为 -100", "lora.md"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", default="")
    p.add_argument("--reranker", default="")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp:
        store = Store(Path(temp)/"eval.sqlite")
        for path in (root/"examples/knowledge").glob("*.md"):
            doc_id, chunks = chunk_document(path.name, path.read_bytes())
            store.replace_document(doc_id, chunks)
        retriever = Retriever(store, args.embedding, args.reranker)
        retriever.search("warmup", 3)
        results = []
        for query, expected in CASES:
            start = time.perf_counter()
            hits = retriever.search(query, 3)
            ranked = [hit["title"] for hit in hits]
            rank = ranked.index(expected)+1 if expected in ranked else None
            results.append({"query": query, "expected": expected, "retrieved": ranked,
                            "rank": rank, "latency_ms": (time.perf_counter()-start)*1000})
        report = {"mode": retriever.mode, "embedding_model": args.embedding, "reranker_model": args.reranker,
                  "cases": len(results), "recall_at_3": sum(r["rank"] is not None for r in results)/len(results),
                  "mrr_at_3": sum(1/r["rank"] if r["rank"] else 0 for r in results)/len(results),
                  "median_latency_ms": statistics.median(r["latency_ms"] for r in results), "results": results,
                  "scope": "12 authored queries against 6 authored documents; document-level recall; no answer-quality claim."}
        name = "rag-dense" if args.embedding else "rag-lexical"
        if args.reranker:
            name += "-reranked"
        path = root/"evidence"/(name+".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
