#!/usr/bin/env python3
"""
03_smoketest_retrieval.py

Quick retrieval sanity test:
- embeds query with BAAI/bge-base-en-v1.5 (normalized)
- searches Milvus (IP / HNSW)
- prints top-k hits with score + key metadata

Usage:
  python stages/03_smoketest_retrieval.py "gravity bomb splash damage"
  python stages/03_smoketest_retrieval.py --topk 8 --ef 96 "hypershot nodes"
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from pymilvus import connections, Collection
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_HOST = "localhost"
DEFAULT_PORT = "19530"
DEFAULT_COLLECTION = "uya_facts"
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"


def connect(host: str, port: str) -> None:
    connections.connect(alias="default", host=host, port=port)


def make_embedder(model_name: str) -> HuggingFaceEmbeddings:
    # normalize_embeddings=True is critical if you're using IP for cosine-ish similarity
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


def pretty_hit(i: int, hit: Any) -> str:
    ent = hit.entity
    text = ent.get("text") or ""
    text_one_line = " ".join(text.split())
    if len(text_one_line) > 220:
        text_one_line = text_one_line[:220] + "…"

    return (
        f"{i:02d}. score={hit.score:.4f}  "
        f"id={ent.get('id')}  "
        f"type={ent.get('type')}  "
        f"category={ent.get('category')}  "
        f"source={ent.get('source_file')}\n"
        f"    {text_one_line}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default="gravity bomb splash damage", help="Query string")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--ef", type=int, default=64, help="HNSW search ef (higher = better recall, slower)")
    ap.add_argument("--metric", default="IP", choices=["IP", "L2", "COSINE"])
    args = ap.parse_args()

    print(f"Connecting Milvus: {args.host}:{args.port}")
    connect(args.host, args.port)

    col = Collection(args.collection)
    col.load()

    print(f"Embedding model: {args.model}")
    embedder = make_embedder(args.model)

    print(f"Query: {args.query!r}")
    q = embedder.embed_query(args.query)

    res = col.search(
        data=[q],
        anns_field="embedding",
        param={"metric_type": args.metric, "params": {"ef": args.ef}},
        limit=args.topk,
        output_fields=["id", "text", "type", "category", "source_file"],
    )

    hits = res[0]
    if not hits:
        print("No results returned.")
        return

    print(f"\nTop {len(hits)} results:\n")
    for i, hit in enumerate(hits, start=1):
        print(pretty_hit(i, hit))
        print()


if __name__ == "__main__":
    main()
