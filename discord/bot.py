from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import discord
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import Collection, connections


ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"
DB_PATH = ROOT_DIR / "infra" / "sqlite" / "discord_bot.sqlite3"
load_dotenv(ROOT_DIR / ".env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN in environment.")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "uya_facts")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RAG_TOPK = int(os.getenv("RAG_TOPK", "12"))
RAG_MAX_CHARS = int(os.getenv("RAG_MAX_CHARS", "6000"))
RAG_METRIC = os.getenv("RAG_METRIC", "IP")
RAG_EF = int(os.getenv("RAG_EF", "96"))

LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8080/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen")
LLM_TIMEOUT_S = int(os.getenv("LLM_TIMEOUT_S", "60"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))


def init_prompt_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_prompts (
                discord_username TEXT PRIMARY KEY,
                systemprompt TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_updates (
                discord_username TEXT NOT NULL,
                created_date TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_saved_prompt(discord_username: str) -> Optional[str]:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT systemprompt FROM user_prompts WHERE discord_username = ?",
            (discord_username,),
        ).fetchone()
    if not row:
        return None
    prompt = (row[0] or "").strip()
    return prompt or None


def save_prompt(discord_username: str, prompt: str) -> None:
    clean_prompt = prompt.strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_prompts (discord_username, systemprompt)
            VALUES (?, ?)
            ON CONFLICT(discord_username) DO UPDATE SET systemprompt = excluded.systemprompt
            """,
            (discord_username, clean_prompt),
        )
        conn.commit()


def clear_prompt(discord_username: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "DELETE FROM user_prompts WHERE discord_username = ?",
            (discord_username,),
        )
        conn.commit()


def save_update(discord_username: str, update_text: str) -> None:
    created_date = datetime.now(timezone.utc).isoformat()
    clean_update = update_text.strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_updates (discord_username, created_date, text)
            VALUES (?, ?, ?)
            """,
            (discord_username, created_date, clean_update),
        )
        conn.commit()


def setup_logger() -> logging.Logger:
    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"discord_bot_{stamp}.log"

    logger = logging.getLogger("discord_bot")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    return logger


LOGGER = setup_logger()

EMBEDDER = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)


init_prompt_db()


def load_prompt_template() -> str:
    if not PROMPT_PATH.exists():
        raise RuntimeError(f"Missing prompt template: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def sanitize_text(text: str) -> str:
    return " ".join(text.split())


def remove_bot_mention(content: str, bot_user_id: int) -> str:
    stripped = content.replace(f"<@{bot_user_id}>", "").replace(
        f"<@!{bot_user_id}>", ""
    )
    return stripped.strip()


def retrieve_rag_context(query: str, k: int = RAG_TOPK) -> str:
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(MILVUS_COLLECTION)
    collection.load()

    q_emb = EMBEDDER.embed_query(query)
    search_res = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": RAG_METRIC, "params": {"ef": RAG_EF}},
        limit=k,
        output_fields=["id", "text", "type", "category", "source_file"],
    )
    hits = search_res[0]
    if not hits:
        return "No retrieval results."

    parts = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        ent = hit.entity
        text = (ent.get("text") or "").strip()
        if not text:
            continue
        block = (
            f"[{i}] score={hit.score:.4f} "
            f"type={ent.get('type')} category={ent.get('category')} source={ent.get('source_file')}\n"
            f"{text}"
        )
        if used + len(block) + 2 > RAG_MAX_CHARS:
            break
        parts.append(block)
        used += len(block) + 2

    if not parts:
        return "No retrieval results."
    return "\n\n".join(parts)


def build_prompt(query: str, rag_context: str, chat_history: str) -> str:
    return build_prompt_with_user_system(query, rag_context, chat_history, "")


def build_prompt_with_user_system(
    query: str, rag_context: str, chat_history: str, user_system_prompt: str
) -> str:
    template = load_prompt_template()
    prompt = (
        template.replace("<< USER SYSTEM PROMPT >>", user_system_prompt)
        .replace("<< RAG INSERTION >>", rag_context)
        .replace("<< CHAT HISTORY >>", chat_history)
    )
    return f"{prompt}\n\n== User Query ==\n{query}"


def call_llm(prompt: str) -> str:
    url = f"{LLM_API_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "n_predict": LLM_MAX_TOKENS,
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=LLM_TIMEOUT_S)
    resp.raise_for_status()
    data = resp.json()

    try:
        content = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception:
        content = ""

    if not content:
        try:
            content = (data["choices"][0].get("text") or "").strip()
        except Exception:
            content = ""

    return content


async def fetch_chat_history(
    channel: discord.abc.Messageable, before_message: discord.Message, limit: int = 10
) -> str:
    rows = []
    async for msg in channel.history(limit=limit, before=before_message):
        text = msg.content.strip()
        if not text:
            if msg.attachments:
                text = "[attachment]"
            else:
                continue
        rows.append(f"{msg.author.display_name}: {sanitize_text(text)}")

    rows.reverse()
    if not rows:
        return "(no prior chat history)"
    return "\n".join(rows)


async def send_chunked(channel: discord.abc.Messageable, text: str) -> None:
    if not text:
        text = "[EMPTY RESPONSE]"
    chunk_size = 1900
    for i in range(0, len(text), chunk_size):
        await channel.send(text[i : i + chunk_size])


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


@client.event
async def on_ready() -> None:
    LOGGER.info("Bot connected as %s", client.user)


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    if not client.user:
        return

    raw_content = message.content.strip()
    if raw_content.startswith("?save"):
        saved_prompt = raw_content[len("?save") :].strip()
        if not saved_prompt:
            await message.channel.send("Usage: `?save [prompt]`")
            return

        username = str(message.author)
        save_prompt(username, saved_prompt)
        LOGGER.info("Saved system prompt for %s: %s", username, saved_prompt)
        await message.channel.send(
            "Thanks! I'll use this in following conversations with you"
        )
        return

    if raw_content == "?clear":
        username = str(message.author)
        clear_prompt(username)
        LOGGER.info("Cleared system prompt for %s", username)
        await message.channel.send("Cleared your saved prompt.")
        return

    if raw_content == "?like":
        await message.channel.send(
            "If you like this, make sure to star https://github.com/Horizon-Private-Server/horizon-help-bot on GitHub!"
        )
        return

    if raw_content.startswith("?update"):
        update_text = raw_content[len("?update") :].strip()
        if not update_text:
            await message.channel.send("Usage: `?update [text]`")
            return

        username = str(message.author)
        save_update(username, update_text)
        LOGGER.info("Saved update from %s: %s", username, update_text)
        await message.channel.send("Saved your update. Thank you.")
        return

    if client.user not in message.mentions:
        return

    user_query = remove_bot_mention(message.content, client.user.id)
    if not user_query:
        user_query = "No query text was provided."

    try:
        chat_history = await fetch_chat_history(message.channel, message, limit=10)
        username = str(message.author)
        saved_user_prompt = get_saved_prompt(username)
        user_system_prompt = ""
        if saved_user_prompt:
            user_system_prompt = (
                f"Follow these additional instructions: {saved_user_prompt}"
            )

        async with message.channel.typing():
            rag_context = await asyncio.to_thread(retrieve_rag_context, user_query, RAG_TOPK)
            llm_prompt = build_prompt_with_user_system(
                user_query, rag_context, chat_history, user_system_prompt
            )
            llm_output = await asyncio.to_thread(call_llm, llm_prompt)

        LOGGER.info("Query from %s: %s", message.author, user_query)
        LOGGER.info("LLM prompt:\n%s", llm_prompt)
        LOGGER.info("LLM output:\n%s", llm_output)

        await send_chunked(message.channel, llm_output)
    except Exception as exc:
        LOGGER.exception("Failed to process message: %s", exc)
        await message.channel.send("Sorry, I hit an error while generating a reply.")


client.run(DISCORD_TOKEN)
