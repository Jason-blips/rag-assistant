import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ask_question import _retrieve, _summarize_with_llm_stream, _doc_source_hint
from vectorstore_utils import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CONVERSATION_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    compute_course_match_score,
    load_vectorstore,
    rerank,
)
from langchain_core.documents import Document


app = FastAPI(title="RAG Chat Backend（流式 SSE）")

#
# 双阈值策略：
# - 触发阈值：低于 0.3 不检索
# - 可用阈值：触发检索后，若 top-k(KB) 最高分仍低于 0.5，则检索内容不喂给 LLM（直接走纯 LLM 生成）
RETRIEVAL_MATCH_GATE = 0.30
RETRIEVAL_USABLE_THRESHOLD = 0.50
LOG_DIR = Path(__file__).resolve().parent / "logs"
ROUTING_LOG_PATH = LOG_DIR / "routing_events.jsonl"
FEEDBACK_LOG_PATH = LOG_DIR / "feedback_events.jsonl"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []
    session_id: str

    k: int = 3
    use_llm: bool = True
    use_reranker: bool = True

    persist_dir: str = str(DEFAULT_PERSIST_DIR)
    collection_name: str = DEFAULT_COLLECTION_NAME
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_NAME
    llm_model: str = "qwen-turbo"


class FeedbackRequest(BaseModel):
    trace_id: str | None = None
    session_id: str
    question: str | None = None
    answer: str
    rating: str  # "up" | "down"
    reason: str | None = None
    route_mode: str | None = None
    match_score: float | None = None
    kb_max_score: float | None = None


def _load_db(
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
):
    return load_vectorstore(
        persist_directory=persist_dir,
        model_name=embedding_model,
        collection_name=collection_name,
    )


def _load_conversation_db(
    persist_dir: str,
    embedding_model: str,
):
    return load_vectorstore(
        persist_directory=persist_dir,
        model_name=embedding_model,
        collection_name=DEFAULT_CONVERSATION_COLLECTION_NAME,
    )


def _get_recent_conversation_docs(
    conv_db,
    session_id: str,
    rounds: int = 3,
) -> list[Document]:
    """
    从对话集合中按 seq 读取最近 N 轮（每轮 user+assistant 两条）历史。
    """
    max_messages = max(1, rounds * 2)
    try:
        raw = conv_db._collection.get(  # noqa: SLF001 - 需要精确按 metadata 读取
            where={"session_id": session_id},
            include=["metadatas", "documents"],
        )
        metadatas = raw.get("metadatas") or []
        documents = raw.get("documents") or []
        docs: list[Document] = []
        for m, d in zip(metadatas, documents):
            md = m or {}
            docs.append(Document(page_content=d, metadata=md))
        docs.sort(key=lambda x: (x.metadata or {}).get("seq", -1))
        return docs[-max_messages:]
    except Exception:
        return []


def _enhance_query_with_conversation(question: str, conv_docs: list[Document]) -> str:
    if not conv_docs:
        return question
    lines: list[str] = []
    for d in conv_docs[-6:]:
        md = d.metadata or {}
        role = md.get("role", "conversation")
        text = (d.page_content or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    if not lines:
        return question
    history_block = "\n".join(lines)
    return f"{question}\n\n[最近对话历史]\n{history_block}"


def _format_docs_for_display(docs, scores: List[float] | None) -> str:
    if not docs:
        return "没有检索到所需内容"

    lines: list[str] = []
    for i, doc in enumerate(docs):
        md = getattr(doc, "metadata", None) or {}
        is_conversation = md.get("doc_type") == "conversation"
        score_str = ""
        if scores is not None and i < len(scores):
            score_str = f" | score={scores[i]:.4f}"
        hint = _doc_source_hint(doc)
        header = (
            f"【对话片段 {i + 1}】{score_str}"
            if is_conversation
            else f"【段落 {i + 1}】{score_str}"
        )
        if hint:
            header += f"\n来源：{hint}"
        lines.append(f"{header}\n{doc.page_content}")
    return "\n\n" + ("-" * 40 + "\n\n").join(lines)


def _sse_event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _append_routing_log(event: Dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(ROUTING_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # 日志写入失败不应影响主流程
        pass


def _append_feedback_log(event: Dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _build_source_suffix(docs, llm_model: str) -> str:
    """
    给最终回答附加来源标识：
    - 有课件检索命中：参考课件段落X-Y
    - 无课件命中（降级）：该问题回答由 <model> 模型生成
    """
    kb_positions: list[int] = []
    for i, d in enumerate(docs or []):
        md = getattr(d, "metadata", None) or {}
        if md.get("doc_type") == "kb":
            kb_positions.append(i + 1)

    if kb_positions:
        if len(kb_positions) == 1:
            return f"参考课件段落{kb_positions[0]}"

        contiguous = all(
            b - a == 1 for a, b in zip(kb_positions, kb_positions[1:])
        )
        if contiguous:
            return f"参考课件段落{kb_positions[0]}-{kb_positions[-1]}"
        return "参考课件段落" + "、".join(str(x) for x in kb_positions)

    return f"该问题回答由 {llm_model} 模型生成"


def _build_source_refs(docs) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for i, d in enumerate(docs or []):
        md = getattr(d, "metadata", None) or {}
        if md.get("doc_type") != "kb":
            continue
        source = md.get("source") or md.get("file_path") or ""
        page = md.get("page")
        hint = _doc_source_hint(d)
        label = f"段落{i + 1}"
        if page is not None:
            try:
                label += f" · p{int(page) + 1}"
            except Exception:
                label += f" · p{page}"

        file_url = ""
        if source:
            src = str(source).replace("\\", "/")
            if src.startswith("/"):
                file_url = f"file://{src}"
            else:
                file_url = f"file:///{src}"
            if page is not None:
                file_url += f"#page={page}"

        refs.append({"label": label, "hint": hint, "url": file_url})
    return refs


def _build_degrade_suggestions(question: str, topic: str) -> list[str]:
    base = [
        "请把问题改写为课件术语（如“UML 类图中的类代表什么”）。",
        "请补充章节名、关键词或页码后再提问。",
        "如果你想要通用解答，可明确写“无需课件依据，给通用答案”。",
    ]
    if topic == "sql":
        return [
            "可尝试：课件中 SQL 的 SELECT / WHERE 示例是什么？",
            "可尝试：课件里 passenger 与 class_table 的 JOIN 如何写？",
            *base,
        ]
    if topic == "uml":
        return [
            "可尝试：课件里 UML association 与 inheritance 的区别是什么？",
            "可尝试：课件中 UML 类图类的定义在哪一页？",
            *base,
        ]
    return base


def _chat_stream(req: ChatRequest) -> Iterator[str]:
    trace_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    match_score, inferred_topic = compute_course_match_score(req.question)
    should_retrieve = match_score >= RETRIEVAL_MATCH_GATE
    route_mode = "retrieval" if should_retrieve else "direct_llm"
    kb_max_score = None
    degrade_reason = None

    docs = []
    scores = None
    kb_db = None
    conv_db = None

    # 1) 只有在“匹配度足够高”时才加载向量库并做检索。
    #    这样满足“低于 30% 不选择查询向量库”的性能诉求。
    if should_retrieve:
        try:
            kb_db = _load_db(
                persist_dir=req.persist_dir,
                collection_name=req.collection_name,
                embedding_model=req.embedding_model,
            )
            conv_db = _load_conversation_db(
                persist_dir=req.persist_dir,
                embedding_model=req.embedding_model,
            )
        except Exception as e:
            _append_routing_log(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "trace_id": trace_id,
                    "stage": "load_db_error",
                    "error": str(e),
                    "question": req.question,
                    "session_id": req.session_id,
                    "match_score": round(match_score, 4),
                    "route_mode": route_mode,
                }
            )
            yield _sse_event(
                {"type": "error", "content": f"加载向量库失败：{e}"}
            )
            return

        # 2) 先把“当前用户问题”写入对话向量库，再进行检索，
        #    这样本轮检索就能把最近对话向量一起纳入 context。
        user_seq = max(len(req.history) - 1, 0)
        try:
            conv_db.add_documents(
                documents=[
                    Document(
                        page_content=req.question,
                        metadata={
                            "doc_type": "conversation",
                            "session_id": req.session_id,
                            "role": "user",
                            "seq": user_seq,
                        },
                    )
                ],
                ids=[f"conv:{req.session_id}:{user_seq}:user"],
            )
        except Exception:
            pass

        try:
            conv_docs = _get_recent_conversation_docs(
                conv_db,
                req.session_id,
                rounds=3,
            )
            enhanced_query = _enhance_query_with_conversation(
                req.question,
                conv_docs,
            )

            kb_docs, kb_scores = _retrieve(
                kb_db,
                enhanced_query,
                k=req.k,
                persist_dir=req.persist_dir,
                use_reranker=req.use_reranker,
            )
        except Exception as e:
            _append_routing_log(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "trace_id": trace_id,
                    "stage": "retrieve_error",
                    "error": str(e),
                    "question": req.question,
                    "session_id": req.session_id,
                    "match_score": round(match_score, 4),
                    "route_mode": route_mode,
                }
            )
            yield _sse_event({"type": "error", "content": f"检索失败：{e}"})
            return

        if kb_scores is not None and kb_docs:
            # 仅以课件向量集合的得分判断是否可用。
            kb_max_score = max(kb_scores)
            if kb_max_score < RETRIEVAL_USABLE_THRESHOLD:
                docs = []
                scores = None
                route_mode = "degraded_low_kb_score"
                degrade_reason = (
                    f"kb_max_score={kb_max_score:.4f} < {RETRIEVAL_USABLE_THRESHOLD:.2f}"
                )
            else:
                # KB 可用时，将“最近 3 轮对话片段 + KB 候选”一起重排，增强上下文关联。
                merged_docs = list(kb_docs)
                for d in conv_docs:
                    md = d.metadata or {}
                    md = {
                        **md,
                        "doc_type": "conversation",
                    }
                    merged_docs.append(Document(page_content=d.page_content, metadata=md))

                if req.use_reranker and merged_docs:
                    reranked_docs, reranked_scores = rerank(
                        query=req.question,
                        docs=merged_docs,
                        top_k=req.k,
                    )
                    # 输出时优先保留课件段落，避免对话片段排到最前造成“像没检索到课件”。
                    selected_docs: list[Document] = []
                    selected_scores: list[float] = []

                    # 先选 KB
                    for d, s in zip(reranked_docs, reranked_scores):
                        md = d.metadata or {}
                        if md.get("doc_type") == "kb":
                            selected_docs.append(d)
                            selected_scores.append(s)
                            if len(selected_docs) >= req.k:
                                break

                    # 不足再补 conversation
                    if len(selected_docs) < req.k:
                        for d, s in zip(reranked_docs, reranked_scores):
                            if d in selected_docs:
                                continue
                            selected_docs.append(d)
                            selected_scores.append(s)
                            if len(selected_docs) >= req.k:
                                break

                    docs, scores = selected_docs, selected_scores
                else:
                    docs = merged_docs[:req.k]
                    scores = kb_scores[: len(docs)]
        else:
            docs = []
            scores = None
            route_mode = "degraded_no_kb_docs"
            degrade_reason = "kb_docs_empty_or_scores_missing"
    else:
        degrade_reason = "match_score_below_gate"

    context_text = _format_docs_for_display(docs, scores)
    source_refs = _build_source_refs(docs)
    degraded = route_mode != "retrieval"
    suggestions = _build_degrade_suggestions(req.question, inferred_topic) if degraded else []
    yield _sse_event(
        {
            "type": "meta",
            "content": {
                "trace_id": trace_id,
                "route_mode": route_mode,
                "match_score": round(match_score, 4),
                "kb_max_score": round(kb_max_score, 4) if kb_max_score is not None else None,
                "source_refs": source_refs,
                "degraded": degraded,
                "suggestions": suggestions,
            },
        }
    )
    yield _sse_event({"type": "context", "content": context_text})

    if not req.use_llm:
        _append_routing_log(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "trace_id": trace_id,
                "question": req.question,
                "session_id": req.session_id,
                "match_score": round(match_score, 4),
                "kb_max_score": round(kb_max_score, 4) if kb_max_score is not None else None,
                "route_mode": route_mode,
                "degrade_reason": degrade_reason,
                "used_llm": False,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            }
        )
        yield _sse_event({"type": "done", "content": ""})
        return

    history_payload = [m.model_dump() for m in req.history]

    try:
        last_text = ""
        for partial in _summarize_with_llm_stream(
            question=req.question,
            docs=docs,
            model=req.llm_model,
            history=history_payload,
        ):
            last_text = partial
            yield _sse_event({"type": "answer", "content": partial})

        # 3) 只有在进行检索时，才把“最终助手回答”写入对话向量库，
        #    避免在非课程/低匹配问题上造成不必要的向量库开销。
        if should_retrieve and conv_db is not None:
            assistant_seq = max(len(req.history) - 1, 0) + 1
            try:
                conv_db.add_documents(
                    documents=[
                        Document(
                            page_content=last_text,
                            metadata={
                                "doc_type": "conversation",
                                "session_id": req.session_id,
                                "role": "assistant",
                                "seq": assistant_seq,
                            },
                        )
                    ],
                    ids=[f"conv:{req.session_id}:{assistant_seq}:assistant"],
                )

                # 只保留最近 3 轮（共 6 条消息）
                keep_last_messages = 6
                min_keep_seq = assistant_seq - (keep_last_messages - 1)
                if min_keep_seq > 0:
                    ids_to_delete: list[str] = []
                    for old_seq in range(min_keep_seq):
                        ids_to_delete.append(f"conv:{req.session_id}:{old_seq}:user")
                        ids_to_delete.append(
                            f"conv:{req.session_id}:{old_seq}:assistant"
                        )
                    try:
                        if ids_to_delete:
                            conv_db.delete(ids=ids_to_delete)
                    except Exception:
                        pass
            except Exception:
                pass

        source_suffix = _build_source_suffix(docs, req.llm_model)
        if last_text.strip():
            final_text = f"{last_text}\n\n—— {source_suffix}"
        else:
            final_text = source_suffix

        # 额外发一帧 answer，保证前端能看到来源标识（即使 done 不刷新主回答）。
        _append_routing_log(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "trace_id": trace_id,
                "question": req.question,
                "session_id": req.session_id,
                "match_score": round(match_score, 4),
                "kb_max_score": round(kb_max_score, 4) if kb_max_score is not None else None,
                "route_mode": route_mode,
                "degrade_reason": degrade_reason,
                "used_llm": True,
                "answer_chars": len(final_text),
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            }
        )
        yield _sse_event({"type": "answer", "content": final_text})
        yield _sse_event({"type": "done", "content": final_text})
    except Exception as e:
        _append_routing_log(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "trace_id": trace_id,
                "stage": "llm_error",
                "error": str(e),
                "question": req.question,
                "session_id": req.session_id,
                "match_score": round(match_score, 4),
                "kb_max_score": round(kb_max_score, 4) if kb_max_score is not None else None,
                "route_mode": route_mode,
                "degrade_reason": degrade_reason,
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            }
        )
        yield _sse_event({"type": "error", "content": f"调用 LLM 失败：{e}"})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _chat_stream(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    if req.rating not in {"up", "down"}:
        return {"ok": False, "error": "rating must be 'up' or 'down'"}
    _append_feedback_log(
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trace_id": req.trace_id,
            "session_id": req.session_id,
            "question": req.question,
            "answer": req.answer,
            "rating": req.rating,
            "reason": req.reason,
            "route_mode": req.route_mode,
            "match_score": req.match_score,
            "kb_max_score": req.kb_max_score,
        }
    )
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

