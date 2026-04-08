"""
corpus.jsonl → spaCy KG → LightRAG
=====================================
用法：
    # Step 1: 只构建 KG，验证结果（不需要 API Key）
    python build_kg_from_corpus.py --mode build --corpus corpus.jsonl

    # Step 2: 构建 + 插入 LightRAG（需要配置 embedding）
    python build_kg_from_corpus.py --mode insert --corpus corpus.jsonl
"""

import re
import json
import asyncio
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spacy_KG_builder import SpacyKGBuilder, CustomKG


# ─────────────────────────────────────────────
# 0. 文本清理
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    清理 tiktoken 不允许的特殊 token。
    语料里含有 <|endoftext|>、<|im_start|> 等时，
    tiktoken 在计算 chunk token 数时会抛 ValueError。
    统一替换所有 <|xxx|> 格式为空字符串。
    """
    return re.sub(r'<\|[^|]+\|>', '', text)


# ─────────────────────────────────────────────
# 1. 读取 corpus.jsonl
# ─────────────────────────────────────────────

def load_corpus(jsonl_path: str) -> list[tuple[str, str]]:
    """
    读取 corpus.jsonl，每行格式：{"id": "...", "text": "...", ...}
    返回 [(doc_id, text), ...]，跳过 text 为空的行。
    """
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] line {lineno} JSON parse error: {e}, skipped.")
                continue
            doc_id = obj.get("id", f"doc_{lineno}")
            text = obj.get("text", "").strip()
            if text:
                docs.append((doc_id, text))
    return docs


# ─────────────────────────────────────────────
# 2. 构建全局 KG
# ─────────────────────────────────────────────

def build_global_kg(
    jsonl_path: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    min_entity_freq: int = 1,
    coref_window: int = 3,
) -> CustomKG:
    builder = SpacyKGBuilder(
        model_name="en_core_web_lg",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_entity_freq=min_entity_freq,
        coref_window=coref_window,
    )

    docs = load_corpus(jsonl_path)
    total = len(docs)
    print(f"Loaded {total} documents from '{jsonl_path}'.")

    global_kg = CustomKG()

    for i, (doc_id, text) in enumerate(docs):
        if i % 20 == 0:
            print(f"  [{i}/{total}] processing '{doc_id}' (len={len(text)}) ...")

        # 清理特殊 token，避免 tiktoken 报错
        text = clean_text(text)

        kg = builder.build_kg_from_text(text, file_path=doc_id)
        global_kg.chunks.extend(kg.chunks)
        global_kg.entities.extend(kg.entities)
        global_kg.relationships.extend(kg.relationships)

    print(f"\n{'='*50}")
    print(f"  Documents    : {total}")
    print(f"  Chunks       : {len(global_kg.chunks)}")
    print(f"  Entities     : {len(global_kg.entities)}")
    print(f"  Relationships: {len(global_kg.relationships)}")
    print(f"{'='*50}")
    return global_kg


# ─────────────────────────────────────────────
# 3. 插入 LightRAG
# ─────────────────────────────────────────────

async def insert_to_lightrag(
    kg: CustomKG,
    working_dir: str = "./rag_storage",
):
    import numpy as np
    from lightrag import LightRAG
    from lightrag.utils import wrap_embedding_func_with_attrs
    from lightrag.llm.openai import openai_complete_if_cache
    from sentence_transformers import SentenceTransformer

    os.makedirs(working_dir, exist_ok=True)

    # ── Embedding ───────────────────────────
    print("Loading Qwen3-Embedding-0.6B ...")
    _embed_model = SentenceTransformer("./models/Qwen3-Embedding-0.6B")
    _embed_model.max_seq_length = 512   # prevent padding to 8192, saves ~16GB VRAM
    print("Embedding model loaded.")

    @wrap_embedding_func_with_attrs(
        embedding_dim=1024,
        max_token_size=8192,
        model_name="Qwen3-Embedding-0.6B",
    )
    async def embed_func(texts: list[str]) -> np.ndarray:
        return np.array(_embed_model.encode(
            texts, normalize_embeddings=True, batch_size=8
        ))

    # ── LLM（查询时使用，插入时不调用）────────
    QWEN_API_KEY  = os.getenv("DASHSCOPE_API_KEY", "YOUR_DASHSCOPE_API_KEY_HERE")
    QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    async def qwen_llm(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            "qwen-plus",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL,
            **kwargs,
        )

    # ── 初始化 LightRAG ─────────────────────
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=embed_func,
        llm_model_func=qwen_llm,
        force_llm_summary_on_merge=999999,
    )
    await rag.initialize_storages()

    # ── 分批插入：每 DOCS_PER_BATCH 篇合并一次，平衡速度与内存 ──
    # 单篇插入调用开销大（锁、IO）；全量插入 OOM。
    # 每批约 20 篇，平均关系数 ~8000，np.concatenate 峰值 ~64 MB。
    # 如果仍 OOM 可调小；如果速度不够可调大（建议上限 50）。
    DOCS_PER_BATCH = 40

    from collections import defaultdict

    doc_chunks:   dict = defaultdict(list)
    doc_entities: dict = defaultdict(list)
    doc_rels:     dict = defaultdict(list)
    for c in kg.chunks:
        doc_chunks[c.file_path].append(c)
    for e in kg.entities:
        doc_entities[e.file_path].append(e)
    for r in kg.relationships:
        doc_rels[r.file_path].append(r)

    all_fps = sorted(
        set(list(doc_chunks.keys()) + list(doc_entities.keys()) + list(doc_rels.keys()))
    )
    total_docs = len(all_fps)
    n_batches  = (total_docs + DOCS_PER_BATCH - 1) // DOCS_PER_BATCH
    print(f"\nInserting KG into LightRAG "
          f"({total_docs} docs, {n_batches} batches of {DOCS_PER_BATCH}) ...")

    for batch_idx in range(n_batches):
        start = batch_idx * DOCS_PER_BATCH
        fps   = all_fps[start : start + DOCS_PER_BATCH]

        batch_kg = CustomKG(
            chunks        = [c for fp in fps for c in doc_chunks.get(fp, [])],
            entities      = [e for fp in fps for e in doc_entities.get(fp, [])],
            relationships = [r for fp in fps for r in doc_rels.get(fp, [])],
        )
        print(f"  Batch {batch_idx + 1}/{n_batches}: "
              f"{len(fps)} docs, {len(batch_kg.relationships)} rels ...")
        await rag.ainsert_custom_kg(batch_kg.to_lightrag_format())

    print("Insert complete.")

    await rag.finalize_storages()
    print(f"Data saved to '{working_dir}'.")


# ─────────────────────────────────────────────
# 3. KG 缓存（保存 / 加载）
# ─────────────────────────────────────────────

def save_kg(kg: "CustomKG", path: str) -> None:
    """把 CustomKG 序列化成 JSON 存到磁盘。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kg.to_lightrag_format(), f, ensure_ascii=False)
    print(f"KG saved to '{path}'  "
          f"(chunks={len(kg.chunks)}, entities={len(kg.entities)}, "
          f"relationships={len(kg.relationships)})")


def load_kg(path: str) -> "CustomKG":
    """从磁盘 JSON 恢复 CustomKG。"""
    from spacy_KG_builder import CustomKG, KGChunk, KGEntity, KGRelationship
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    kg = CustomKG()
    for c in data["chunks"]:
        kg.chunks.append(KGChunk(**c))
    for e in data["entities"]:
        kg.entities.append(KGEntity(**e))
    for r in data["relationships"]:
        kg.relationships.append(KGRelationship(**r))
    print(f"KG loaded from '{path}'  "
          f"(chunks={len(kg.chunks)}, entities={len(kg.entities)}, "
          f"relationships={len(kg.relationships)})")
    return kg


# ─────────────────────────────────────────────
# 4. 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "insert"], default="build")
    parser.add_argument("--corpus", default="corpus.jsonl")
    parser.add_argument("--working_dir", default="./rag_storage")
    parser.add_argument("--chunk_size", type=int, default=1200)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    parser.add_argument("--min_entity_freq", type=int, default=1)
    parser.add_argument("--kg_cache", default="./kg_cache.json",
                        help="KG 缓存文件路径；存在时直接加载，跳过 spaCy 抽取")
    parser.add_argument("--force_rebuild", action="store_true",
                        help="忽略缓存，强制重新抽取 KG")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── 加载或构建 KG ─────────────────────────
    if not args.force_rebuild and os.path.exists(args.kg_cache):
        kg = load_kg(args.kg_cache)
    else:
        kg = build_global_kg(
            jsonl_path=args.corpus,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            min_entity_freq=args.min_entity_freq,
        )
        save_kg(kg, args.kg_cache)

    # ── 插入 LightRAG ─────────────────────────
    if args.mode == "insert":
        asyncio.run(insert_to_lightrag(
            kg=kg,
            working_dir=args.working_dir,
        ))
    else:
        print("\nMode=build: KG built and cached, NOT inserted into LightRAG.")
        print("Run with --mode insert to insert.")