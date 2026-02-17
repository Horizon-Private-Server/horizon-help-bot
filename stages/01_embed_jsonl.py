#!/usr/bin/env python3
"""
01_embed_jsonl.py

Embed a JSONL corpus (one object per line with at least {"id": ..., "text": ...})
using BGE base (BAAI/bge-base-en-v1.5) via LangChain's HuggingFaceEmbeddings.

Outputs:
- <output_dir>/<basename>.embeddings.npy    float32 matrix (N x 768)
- <output_dir>/<basename>.ids.txt           ids aligned to embeddings row order
- <output_dir>/<basename>.embedded.jsonl    each line includes {"id","embedding", ...metadata}
- <output_dir>/<basename>.manifest.json     run metadata and provenance

Example:
  python 01_embed_jsonl.py --input output/vector_ready.jsonl --output-dir output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings


EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EXPECTED_DIM = 768
DEFAULT_INPUT_JSONL = "output/vector_ready.jsonl"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_OUTPUT_BASENAME = "vector_ready"


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    """Yield (line_no, obj) from a JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e


def build_embedder(device: str) -> HuggingFaceEmbeddings:
    """
    Build BGE embedder. normalize_embeddings=True means cosine and IP are equivalent if vectors are normalized.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_record(line_no: int, rec: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    """
    Ensure record has required fields and return (id, text, metadata_without_text).
    Metadata is everything except 'text' and (optionally) pre-existing 'embedding' fields.
    """
    rec_id = rec.get("id")
    text = rec.get("text")

    if not isinstance(rec_id, str) or not rec_id.strip():
        raise ValueError(f"Line {line_no}: missing/invalid 'id' (expected non-empty string).")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Line {line_no} (id={rec_id}): missing/invalid 'text' (expected non-empty string).")

    # Keep all other fields as metadata; drop any existing 'embedding' to avoid confusion
    meta = {k: v for k, v in rec.items() if k not in ("text", "embedding")}
    return rec_id, text, meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed a JSONL corpus with BGE base via LangChain.")
    ap.add_argument(
        "--input",
        "--in",
        dest="in_path",
        default=DEFAULT_INPUT_JSONL,
        required=False,
        help="Input JSONL path (default: %(default)s)",
    )
    ap.add_argument(
        "--output-dir",
        "--out",
        dest="out_dir",
        default=DEFAULT_OUTPUT_DIR,
        required=False,
        help="Output directory (default: %(default)s)",
    )
    ap.add_argument(
        "--basename",
        default=DEFAULT_OUTPUT_BASENAME,
        required=False,
        help="Output file basename (default: %(default)s)",
    )
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for embedding")
    ap.add_argument("--max", type=int, default=0, help="Max records to process (0 = all)")
    ap.add_argument("--include-text", action="store_true", help="Include original text in embedded.jsonl")
    ap.add_argument(
        "--embedding-field",
        default="embedding",
        help="Field name to store vectors in embedded.jsonl (default: embedding)",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"Input file does not exist: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record provenance
    basename = args.basename
    manifest_path = out_dir / f"{basename}.manifest.json"
    manifest: Dict[str, Any] = {
        "input_path": str(in_path),
        "input_sha256": sha256_of_file(in_path),
        "model": EMBED_MODEL_NAME,
        "expected_dim": EXPECTED_DIM,
        "device": args.device,
        "batch": args.batch,
        "max": args.max,
        "include_text": bool(args.include_text),
        "embedding_field": args.embedding_field,
        "cwd": os.getcwd(),
    }

    embedder = build_embedder(args.device)

    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    batch_ids: List[str] = []
    batch_texts: List[str] = []
    batch_metas: List[Dict[str, Any]] = []

    vectors: List[List[float]] = []

    processed = 0
    for line_no, rec in iter_jsonl(in_path):
        rec_id, text, meta = validate_record(line_no, rec)

        batch_ids.append(rec_id)
        batch_texts.append(text)
        batch_metas.append(meta)

        if len(batch_texts) >= args.batch:
            batch_vecs = embedder.embed_documents(batch_texts)

            ids.extend(batch_ids)
            texts.extend(batch_texts)
            metas.extend(batch_metas)
            vectors.extend(batch_vecs)

            processed += len(batch_texts)
            if processed % (args.batch * 10) == 0:
                eprint(f"[embed] processed {processed} records...")

            batch_ids, batch_texts, batch_metas = [], [], []

            if args.max and processed >= args.max:
                break

    # Flush remainder
    if batch_texts and (not args.max or processed < args.max):
        batch_vecs = embedder.embed_documents(batch_texts)

        ids.extend(batch_ids)
        texts.extend(batch_texts)
        metas.extend(batch_metas)
        vectors.extend(batch_vecs)

        processed += len(batch_texts)

    if processed == 0:
        raise SystemExit("No records embedded (empty input?)")

    emb = np.asarray(vectors, dtype=np.float32)

    if emb.ndim != 2 or emb.shape[1] != EXPECTED_DIM:
        raise SystemExit(f"Unexpected embedding shape {emb.shape}. Expected N x {EXPECTED_DIM}.")

    # Save matrix + ids
    npy_path = out_dir / f"{basename}.embeddings.npy"
    ids_path = out_dir / f"{basename}.ids.txt"
    np.save(npy_path, emb)
    ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")

    # Save embedded JSONL
    embedded_jsonl_path = out_dir / f"{basename}.embedded.jsonl"
    with embedded_jsonl_path.open("w", encoding="utf-8") as f:
        for rec_id, vec, meta, text in zip(ids, vectors, metas, texts):
            out_obj: Dict[str, Any] = {"id": rec_id, args.embedding_field: vec, **meta}
            if args.include_text:
                out_obj["text"] = text
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    # Finalize manifest
    manifest.update(
        {
            "num_records": int(emb.shape[0]),
            "embedding_dim": int(emb.shape[1]),
            "outputs": {
                "embeddings_npy": str(npy_path),
                "ids_txt": str(ids_path),
                "embedded_jsonl": str(embedded_jsonl_path),
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Embedded {emb.shape[0]} records -> {emb.shape}")
    print(f"Wrote {npy_path}")
    print(f"Wrote {ids_path}")
    print(f"Wrote {embedded_jsonl_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
