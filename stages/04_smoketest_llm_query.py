#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pymilvus import connections, Collection
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import requests

COLLECTION_NAME = "uya_facts"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# OpenAI-compatible llama.cpp server base (no trailing slash)
LLM_BASE_URL = "http://172.31.222.51:8080/v1"
LLM_MODEL = "qwen"

DEFAULT_TOPK = 12
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT_S = 60


def connect_milvus() -> None:
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def retrieve(query: str, k: int):
    col = Collection(COLLECTION_NAME)
    col.load()

    embedder = get_embedder()
    q_emb = embedder.embed_query(query)

    res = col.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 96}},
        limit=k,
        output_fields=["id", "text", "type", "category", "source_file"],
    )
    return res[0]


def build_context(hits, max_chars: int = 6000) -> str:
    """
    Build compact context with actual fact text.
    Hard cap by characters to avoid blowing the model context window.
    """
    parts = []
    used = 0

    for i, h in enumerate(hits, start=1):
        ent = h.entity
        text = (ent.get("text") or "").strip()
        if not text:
            continue

        block = (
            f"[{i}] score={h.score:.4f} "
            f"type={ent.get('type')} category={ent.get('category')} source={ent.get('source_file')}\n"
            f"{text}"
        )

        if used + len(block) + 2 > max_chars:
            break

        parts.append(block)
        used += len(block) + 2

    return "\n\n".join(parts)


def build_messages(query: str, context: str):
    system = (
        "You are Horizon Help Bot, an expert on Ratchet & Clank: Up Your Arsenal multiplayer.\n"
        "Use ONLY the provided context.\n"
        "If asked for 'best'/'most effective'/'most': interpret as 'most supported by context for the requested property'.\n"
        "If context is insufficient, say exactly: \"I don't know based on the provided context.\".\n"
        "Write a short answer (1–3 sentences).\n"
        "Cite sources inline using the bracket numbers, e.g. [1] or [1][4].\n"
        "Do not cite anything not in context.\n"
    )

    user = (
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

    lc_messages = [SystemMessage(content=system), HumanMessage(content=user)]
    raw_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return lc_messages, raw_messages


def call_llm_langchain(lc_messages, max_tokens: int) -> str:
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key="dummy",
        model=LLM_MODEL,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    resp = llm.invoke(lc_messages)
    return (resp.content or "").strip()


def call_llm_raw_chat(payload: dict, timeout_s: int) -> tuple[str, dict]:
    """
    Calls OpenAI-style /chat/completions and returns (content, raw_json).
    Also attempts fallback extraction if server uses different fields.
    """
    url = f"{LLM_BASE_URL}/chat/completions"
    r = requests.post(url, json=payload, timeout=timeout_s)

    try:
        j = r.json()
    except Exception:
        j = {"_non_json": True, "status": r.status_code, "text": r.text}

    # Attempt OpenAI chat format
    content = ""
    try:
        content = (j["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        pass

    # Fallback: some servers return completions-style "text"
    if not content:
        try:
            content = (j["choices"][0].get("text") or "").strip()
        except Exception:
            pass

    return content, {"status": r.status_code, "headers": dict(r.headers), "json": j, "url": url}


def call_llm_raw_completions(prompt: str, max_tokens: int, timeout_s: int) -> tuple[str, dict]:
    """
    Calls OpenAI-style /completions (prompt-based) and returns (text, raw_json).
    Useful if your server doesn't implement chat completions correctly.
    """
    url = f"{LLM_BASE_URL}/completions"
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "n_predict": max_tokens,
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)

    try:
        j = r.json()
    except Exception:
        j = {"_non_json": True, "status": r.status_code, "text": r.text}

    text = ""
    try:
        text = (j["choices"][0].get("text") or "").strip()
    except Exception:
        pass

    return text, {"status": r.status_code, "headers": dict(r.headers), "json": j, "url": url, "payload": payload}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument(
        "--mode",
        choices=["raw_chat", "raw_completions", "langchain"],
        default="raw_chat",
        help="Which LLM call path to use (raw_chat recommended for debugging).",
    )
    args = ap.parse_args()

    connect_milvus()

    hits = retrieve(args.query, args.topk)
    context = build_context(hits)

    if args.show_context:
        print("\n=== RETRIEVED CONTEXT ===\n")
        print(context)
        print()

    lc_messages, raw_messages = build_messages(args.query, context)

    # ALWAYS SHOW CHAT PAYLOAD (OpenAI chat format)
    chat_payload = {
        "model": LLM_MODEL,
        "messages": raw_messages,
        "temperature": 0.2,
        "max_tokens": args.max_tokens,
        "n_predict": args.max_tokens,  # llama.cpp often honors this
        "stream": False,
    }

    print("\n=== LLM PAYLOAD (chat/completions) ===\n")
    print(json.dumps(chat_payload, indent=2)[:12000])
    print()

    answer = ""
    if args.mode == "langchain":
        answer = call_llm_langchain(lc_messages, max_tokens=args.max_tokens)
        print("\n=== LANGCHAIN RAW ANSWER ===\n")
        print(answer if answer else "[EMPTY RESPONSE]")
    elif args.mode == "raw_chat":
        answer, raw = call_llm_raw_chat(chat_payload, timeout_s=args.timeout)

        print("\n=== LLM RAW HTTP/JSON (chat/completions) ===\n")
        out = {
            "url": raw["url"],
            "status": raw["status"],
            "content_type": raw["headers"].get("Content-Type") or raw["headers"].get("content-type"),
            "json": raw["json"],
        }
        print(json.dumps(out, indent=2)[:12000])
        print()

        if not answer:
            # As a safety net, try prompt-based completions using the user message as prompt
            prompt = raw_messages[-1]["content"]
            answer2, raw2 = call_llm_raw_completions(prompt, max_tokens=args.max_tokens, timeout_s=args.timeout)

            print("\n=== FALLBACK RAW HTTP/JSON (/completions) ===\n")
            out2 = {
                "url": raw2["url"],
                "status": raw2["status"],
                "content_type": raw2["headers"].get("Content-Type") or raw2["headers"].get("content-type"),
                "json": raw2["json"],
                "payload": raw2["payload"],
            }
            print(json.dumps(out2, indent=2)[:12000])
            print()

            if answer2:
                answer = answer2
    else:  # raw_completions
        prompt = raw_messages[-1]["content"]
        answer, raw = call_llm_raw_completions(prompt, max_tokens=args.max_tokens, timeout_s=args.timeout)

        print("\n=== LLM RAW HTTP/JSON (/completions) ===\n")
        out = {
            "url": raw["url"],
            "status": raw["status"],
            "content_type": raw["headers"].get("Content-Type") or raw["headers"].get("content-type"),
            "json": raw["json"],
            "payload": raw["payload"],
        }
        print(json.dumps(out, indent=2)[:12000])
        print()

    print("\n=== ANSWER ===\n")
    print(answer if answer else "[EMPTY RESPONSE]")

    print("\n=== SOURCES ===\n")
    for i, h in enumerate(hits, start=1):
        ent = h.entity
        print(
            f"[{i}] score={h.score:.4f} | "
            f"{ent.get('category')} | {ent.get('type')} | {ent.get('source_file')}"
        )


if __name__ == "__main__":
    main()
