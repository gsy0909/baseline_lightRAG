"""
retrieve_and_eval.py
====================
对 queries.jsonl 里的每个问题做 LightRAG 检索，
记录每个问题检索到的文档，并计算 Recall@K。

用法：
    python retrieve_and_eval.py \
        --queries  toy_dataset_10/toy_dataset_10/queries.jsonl \
        --relevance toy_dataset_10/toy_dataset_10/relevance.jsonl \
        --output   retrieval_results.jsonl \
        --top_k    10

环境变量：
    export OPENAI_API_KEY="sk-..."
"""

import json
import asyncio
import argparse
import os
import numpy as np
from collections import defaultdict
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# 0. OpenAI API 配置
# ─────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
OPENAI_BASE_URL = "https://www.packyapi.com/v1"
LLM_MODEL = "gpt-5.4-high"


async def gpt_llm(
    prompt,
    system_prompt=None,
    history_messages=None,
    **kwargs,
) -> str:
    """
    LLM 函数，指向 GPT-5.4。
    在 only_need_context=True 模式下不会被实际调用，
    但 LightRAG 初始化时必须传入一个合法的 llm_model_func。
    """
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        **kwargs,
    )


# ─────────────────────────────────────────────
# 1. Embedding（必须和插入时一致）
# ─────────────────────────────────────────────

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
    embeddings = _embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=8,
    )
    return np.array(embeddings)


# ─────────────────────────────────────────────
# 2. 读取文件
# ─────────────────────────────────────────────

def load_queries(path: str) -> list[dict]:
    """读取 queries.jsonl，返回 [{"id": ..., "text": ...}, ...]"""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries from '{path}'.")
    return queries


def load_relevance(path: str) -> dict[str, set[str]]:
    """
    读取 relevance.jsonl，返回
    {query_id: {corpus_id1, corpus_id2, ...}}
    只保留 relevance=1 的条目。
    """
    rel = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("relevance", 0) >= 1:
                rel[obj["query-id"]].add(obj["corpus-id"])
    print(f"Loaded relevance for {len(rel)} queries from '{path}'.")
    return dict(rel)


# ─────────────────────────────────────────────
# 3. 检索
# ─────────────────────────────────────────────

async def retrieve_one(
    rag: LightRAG,
    query_id: str,
    query_text: str,
    top_k: int,
    mode: str,
    debug_output: str | None = None,
) -> dict:
    """
    使用 aquery_data 做检索：走完 LightRAG 完整的 KG + 向量检索流程，
    返回结构化数据（entities / relationships / chunks），每条都带 file_path。
    不调用 LLM 生成最终答案，只做检索。
    """
    try:
        result = await rag.aquery_data(
            query_text,
            param=QueryParam(
                mode=mode,
                top_k=top_k,           # entities/relationships 各取 top_k 个
                chunk_top_k=top_k * 2, # chunks 多取一倍，应对同文档多 chunk 去重
            ),
        )

        # 按优先级合并三个来源：chunks 优先（最直接），再是 entities，再是 relationships
        # 去重后严格截到 top_k 个唯一文档
        seen_set: set[str] = set()
        ordered_docs: list[str] = []

        if result.get("status") == "success":
            data = result.get("data", {})
            for source in ["chunks", "entities", "relationships"]:
                for item in data.get(source, []):
                    # entities/relationships 的 file_path 可能是多文档合并的
                    # 格式为 "doc1.txt<SEP>doc2.txt"，需要拆分
                    fp_raw = item.get("file_path", "")
                    fps = [
                        fp.strip()
                        for fp in fp_raw.split("<SEP>")
                        if fp.strip() and fp.strip() != "unknown"
                    ]
                    for fp in fps:
                        if fp not in seen_set:
                            seen_set.add(fp)
                            ordered_docs.append(fp)
            ordered_docs = ordered_docs[:top_k]

        retrieved_docs = set(ordered_docs)
        print(f"[{query_id}] retrieved {len(retrieved_docs)} docs: {sorted(retrieved_docs)}")

        # 调试：把完整检索结果写到文件
        if debug_output and result.get("status") == "success":
            debug_data = {
                "query_id":   query_id,
                "query_text": query_text,
                "mode":       mode,
                "top_k":      top_k,
                "chunks":     result.get("data", {}).get("chunks", []),
                "entities":   result.get("data", {}).get("entities", []),
                "relationships": result.get("data", {}).get("relationships", []),
                "ordered_docs": ordered_docs,
            }
            with open(debug_output, "w", encoding="utf-8") as dbf:
                json.dump(debug_data, dbf, ensure_ascii=False, indent=2)
            print(f"  [debug] full retrieval data saved to '{debug_output}'")

        return {
            "query_id":        query_id,
            "retrieved_docs":  list(retrieved_docs),
            "retrieved_count": len(retrieved_docs),
        }

    except Exception as e:
        print(f"  [warn] query '{query_id}' failed: {e}")
        import traceback; traceback.print_exc()
        return {
            "query_id":        query_id,
            "retrieved_docs":  [],
            "retrieved_count": 0,
        }



async def retrieve_all(
    queries: list[dict],
    working_dir: str,
    top_k: int,
    mode: str,
    debug_query_id: str | None = None,
    debug_output: str = "debug_retrieval.json",
) -> list[dict]:
    """对所有问题做检索"""

    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=embed_func,
        llm_model_func=gpt_llm,    # GPT-5.4（only_need_context 时不实际调用）
    )
    await rag.initialize_storages()

    results = []
    total   = len(queries)

    for i, q in enumerate(queries):
        if i % 10 == 0:
            print(f"  [{i}/{total}] retrieving '{q['id']}' ...")

        res = await retrieve_one(
            rag=rag,
            query_id=q["id"],
            query_text=q["text"],
            top_k=top_k,
            mode=mode,
            debug_output=debug_output if q["id"] == debug_query_id else None,
        )
        results.append(res)

    await rag.finalize_storages()
    return results


# ─────────────────────────────────────────────
# 4. Recall 计算
# ─────────────────────────────────────────────

def compute_recall(
    results: list[dict],
    relevance: dict[str, set[str]],
) -> dict:
    """
    计算 Recall：
      Recall(q) = |retrieved(q) ∩ relevant(q)| / |relevant(q)|
    宏平均（macro avg）：所有 query 的 Recall 取平均。
    """
    recall_scores = []
    per_query     = []

    for res in results:
        qid       = res["query_id"]
        retrieved = set(res["retrieved_docs"])
        relevant  = relevance.get(qid, set())

        if not relevant:
            continue

        hits   = retrieved & relevant
        recall = len(hits) / len(relevant)
        recall_scores.append(recall)

        per_query.append({
            "query_id":        qid,
            "recall":          round(recall, 4),
            "hits":            len(hits),
            "relevant_total":  len(relevant),
            "retrieved_total": len(retrieved),
            "hit_docs":        list(hits),
            "missed_docs":     list(relevant - retrieved),
        })

    macro_recall = (
        sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    )

    return {
        "macro_recall":          round(macro_recall, 4),
        "num_queries_evaluated": len(recall_scores),
        "per_query":             per_query,
    }


# ─────────────────────────────────────────────
# 5. 入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries",     required=True,
                        help="queries.jsonl 路径")
    parser.add_argument("--relevance",   required=True,
                        help="relevance.jsonl 路径")
    parser.add_argument("--output",      default="retrieval_results.jsonl",
                        help="检索结果输出路径")
    parser.add_argument("--working_dir", default="./rag_storage",
                        help="LightRAG 数据目录，必须和插入时一致")
    parser.add_argument("--top_k",       type=int, default=10,
                        help="每个问题最多检索多少条结果")
    parser.add_argument("--mode",        default="hybrid",
                        choices=["local", "global", "hybrid", "mix", "naive"],
                        help="LightRAG 检索模式")
    parser.add_argument("--debug_query_id", default=None,
                        help="指定某个 query_id，将其完整检索结果（chunks/entities/relationships）写到 debug_output 文件")
    parser.add_argument("--debug_output",   default="debug_retrieval.json",
                        help="调试输出文件路径（默认 debug_retrieval.json）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    queries   = load_queries(args.queries)
    relevance = load_relevance(args.relevance)

    print(f"\nRetrieving with mode='{args.mode}', top_k={args.top_k} ...")
    results = asyncio.run(retrieve_all(
        queries=queries,
        working_dir=args.working_dir,
        top_k=args.top_k,
        mode=args.mode,
        debug_query_id=args.debug_query_id,
        debug_output=args.debug_output,
    ))

    eval_result = compute_recall(results, relevance)

    print(f"\n{'='*50}")
    print(f"  Macro Recall : {eval_result['macro_recall']:.4f}")
    print(f"  Queries eval : {eval_result['num_queries_evaluated']}")
    print(f"{'='*50}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "summary": {
                "macro_recall":          eval_result["macro_recall"],
                "num_queries_evaluated": eval_result["num_queries_evaluated"],
                "mode":                  args.mode,
                "top_k":                 args.top_k,
            }
        }, ensure_ascii=False) + "\n")
        for pq in eval_result["per_query"]:
            f.write(json.dumps(pq, ensure_ascii=False) + "\n")

    print(f"Results saved to '{args.output}'.")
