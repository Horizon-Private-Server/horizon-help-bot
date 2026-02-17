#!/usr/bin/env python3
"""
02_milvus_upsert.py (Option A)

Hard-refresh Milvus with latest embedded corpus in output/:
- Drops target collection if it exists
- Recreates schema + vector index
- Inserts records by joining:
    output/vector_ready.embedded.jsonl  (id + embedding + thin metadata)
    output/vector_ready.jsonl           (id + text + rich metadata)
    output/vector_ready.embeddings.npy  (optional; used for speed/compactness)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


DEFAULT_COLLECTION = "uya_facts"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = "19530"

OUTPUT_DIR = Path("output")
BASE_JSONL = OUTPUT_DIR / "vector_ready.jsonl"                 # <--- NEW
EMBEDDED_JSONL = OUTPUT_DIR / "vector_ready.embedded.jsonl"
EMBEDDINGS_NPY = OUTPUT_DIR / "vector_ready.embeddings.npy"
MANIFEST_JSON = OUTPUT_DIR / "vector_ready.manifest.json"


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def load_manifest(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        die(f"Failed to read manifest {path}: {e}")
    return None


def connect(host: str, port: str, alias: str = "default") -> None:
    connections.connect(alias=alias, host=host, port=port)


def drop_collection_if_exists(name: str) -> None:
    if utility.has_collection(name):
        utility.drop_collection(name)


def create_collection(name: str, dim: int, text_max_len: int = 8192) -> Collection:
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=128),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=text_max_len),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
    ]
    schema = CollectionSchema(fields=fields, description="UYA Horizon Help Bot fact corpus")
    return Collection(name=name, schema=schema)


def create_index(collection: Collection, index_type: str, metric_type: str) -> None:
    if index_type.upper() == "HNSW":
        index_params = {"index_type": "HNSW", "metric_type": metric_type, "params": {"M": 16, "efConstruction": 200}}
    elif index_type.upper() == "IVF_FLAT":
        index_params = {"index_type": "IVF_FLAT", "metric_type": metric_type, "params": {"nlist": 2048}}
    elif index_type.upper() == "FLAT":
        index_params = {"index_type": "FLAT", "metric_type": metric_type, "params": {}}
    else:
        die(f"Unsupported index type: {index_type}")

    collection.create_index(field_name="embedding", index_params=index_params)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        die(f"Missing file: {path}")
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as e:
                die(f"JSON parse error in {path} line {i}: {e}")
    return records


# -------- NEW: build id -> record map for base jsonl --------
def load_base_map(path: Path) -> Dict[str, Dict[str, Any]]:
    base_records = read_jsonl(path)
    m: Dict[str, Dict[str, Any]] = {}
    for r in base_records:
        rid = r.get("id")
        if not isinstance(rid, str) or not rid:
            die("Found base jsonl record without a valid string 'id'")
        if rid in m:
            die(f"Duplicate id in base jsonl: {rid}")
        m[rid] = r
    return m
# -----------------------------------------------------------


def load_embeddings(path: Path) -> np.ndarray:
    if not path.exists():
        die(f"Missing file: {path}")
    arr = np.load(path)
    if arr.ndim != 2:
        die(f"Embeddings array must be 2D; got shape {arr.shape}")
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float32)
    return arr


def sanity_check(records: List[Dict[str, Any]], emb: np.ndarray, manifest: Dict[str, Any] | None) -> Tuple[int, int]:
    n = len(records)
    if emb.shape[0] != n:
        die(f"Count mismatch: {n} embedded jsonl records vs {emb.shape[0]} embeddings.npy rows")

    dim = emb.shape[1]

    if manifest is not None:
        expected_dim = None
        expected_n = None

        for k in ("embedding_dim", "dim", "vector_dim"):
            if k in manifest:
                expected_dim = manifest[k]
                break

        for k in ("num_records", "count", "n_records", "rows"):
            if k in manifest:
                expected_n = manifest[k]
                break

        if expected_dim is not None and int(expected_dim) != int(dim):
            die(f"Manifest dim {expected_dim} != embeddings dim {dim}")

        if expected_n is not None and int(expected_n) != int(n):
            die(f"Manifest count {expected_n} != records count {n}")

    return n, dim


def batch_insert(
    collection: Collection,
    embedded_records: List[Dict[str, Any]],
    embeddings: np.ndarray,
    base_map: Dict[str, Dict[str, Any]],
    batch_size: int,
) -> None:
    total = len(embedded_records)
    if total == 0:
        die("No records to insert (embedded jsonl is empty).")

    batches = math.ceil(total / batch_size)

    seen_ids: set[str] = set()

    for b in range(batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, total)

        chunk = embedded_records[start:end]
        emb_chunk = embeddings[start:end].tolist()  # aligned by embedded_records order

        ids: List[str] = []
        texts: List[str] = []
        types: List[str] = []
        cats: List[str] = []
        sources: List[str] = []

        for r in chunk:
            rid = r.get("id")
            if not isinstance(rid, str) or not rid:
                die("Found embedded jsonl record without a valid string 'id'")
            if rid in seen_ids:
                die(f"Duplicate id in embedded jsonl: {rid}")
            seen_ids.add(rid)

            base = base_map.get(rid)
            if base is None:
                die(f"Embedded record id not found in base jsonl: {rid}")

            # Backfill text from base jsonl (authoritative)
            text = base.get("text")
            if not isinstance(text, str) or not text.strip():
                die(f"Missing/empty text for id={rid} in base jsonl")
            text = text.strip()

            # Prefer base metadata (more complete), but fallback to embedded record if absent
            wtype = base.get("type") if isinstance(base.get("type"), str) else r.get("type", "")
            cat = base.get("category") if isinstance(base.get("category"), str) else r.get("category", "")
            src = base.get("source_file") if isinstance(base.get("source_file"), str) else r.get("source_file", "")

            ids.append(rid)
            texts.append(text)
            types.append(wtype or "")
            cats.append(cat or "")
            sources.append(src or "")

        data = [ids, emb_chunk, texts, types, cats, sources]
        collection.insert(data)

        print(f"Inserted batch {b+1}/{batches} ({end}/{total})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--index-type", default="HNSW", choices=["HNSW", "IVF_FLAT", "FLAT"])
    ap.add_argument("--metric", default="IP", choices=["IP", "L2", "COSINE"])
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--no-drop", action="store_true", help="Do not drop collection; append/upsert (not recommended).")
    args = ap.parse_args()

    print(f"Connecting to Milvus at {args.host}:{args.port} …")
    connect(args.host, args.port)

    print("Loading base corpus (for text) …")
    base_map = load_base_map(BASE_JSONL)
    print(f"Base records: {len(base_map)}")

    print("Loading embedded corpus …")
    embedded_records = read_jsonl(EMBEDDED_JSONL)
    embeddings = load_embeddings(EMBEDDINGS_NPY)
    manifest = load_manifest(MANIFEST_JSON)

    n, dim = sanity_check(embedded_records, embeddings, manifest)
    print(f"Embedded records: {n} | Embedding dim: {dim}")

    if not args.no_drop:
        print(f"Dropping collection '{args.collection}' (hard refresh) …")
        drop_collection_if_exists(args.collection)
    else:
        if not utility.has_collection(args.collection):
            die(f"--no-drop specified but collection '{args.collection}' does not exist.")
        print(f"--no-drop set: will insert into existing collection '{args.collection}'")

    if not utility.has_collection(args.collection):
        print(f"Creating collection '{args.collection}' …")
        col = create_collection(args.collection, dim=dim)
        print(f"Creating index ({args.index_type}, metric={args.metric}) …")
        create_index(col, index_type=args.index_type, metric_type=args.metric)
    else:
        col = Collection(args.collection)

    print(f"Inserting into '{args.collection}' …")
    batch_insert(col, embedded_records, embeddings, base_map, batch_size=args.batch_size)

    print("Flushing …")
    col.flush()

    print("Loading collection into memory …")
    col.load()

    try:
        count = col.num_entities
        print(f"✅ Done. Collection '{args.collection}' now has {count} entities.")
    except Exception:
        print("✅ Done. (Count unavailable, but insert completed.)")


if __name__ == "__main__":
    main()
