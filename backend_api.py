import json
import os
from typing import Any, Dict, Iterator, List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ask_question import _retrieve, _summarize_with_llm_stream, _doc_source_hint
from vectorstore_utils import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    compute_course_match_score,
    load_vectorstore,
)
from langchain_core.documents import Document


app = FastAPI(title="RAG Chat Backend（流式 SSE）")

# 当检索返回的相关性分数整体很低时，视为“没有检索到所需内容”，
# 但仍然允许 LLM 在空/弱上下文下生成回答（避免直接中止）。
MIN_RELEVANCE_SCORE = 0.05

# 用户要求的“检索触发门槛”：低于 30% 不走向量检索。
RETRIEVAL_MATCH_GATE = 0.30


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


def _chat_stream(req: ChatRequest) -> Iterator[str]:
    match_score, _ = compute_course_match_score(req.question)
    should_retrieve = match_score >= RETRIEVAL_MATCH_GATE

    docs = []
    scores = None
    db = None

    # 1) 只有在“匹配度足够高”时才加载向量库并做检索。
    #    这样满足“低于 30% 不选择查询向量库”的性能诉求。
    if should_retrieve:
        try:
            db = _load_db(
                persist_dir=req.persist_dir,
                collection_name=req.collection_name,
                embedding_model=req.embedding_model,
            )
        except Exception as e:
            yield _sse_event(
                {"type": "error", "content": f"加载向量库失败：{e}"}
            )
            return

        # 2) 先把“当前用户问题”写入对话向量库，再进行检索，
        #    这样本轮检索就能把最近对话向量一起纳入 context。
        user_seq = max(len(req.history) - 1, 0)
        try:
            db.add_documents(
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
            docs, scores = _retrieve(
                db,
                req.question,
                k=req.k,
                persist_dir=req.persist_dir,
                use_reranker=req.use_reranker,
            )
        except Exception as e:
            yield _sse_event({"type": "error", "content": f"检索失败：{e}"})
            return

        if scores is not None and docs:
            # 仅基于分数做轻量判断：分数太低时将 docs 置空，从而触发“没有检索到所需内容”的提示。
            if max(scores) < MIN_RELEVANCE_SCORE:
                docs = []
                scores = None

    context_text = _format_docs_for_display(docs, scores)
    yield _sse_event({"type": "context", "content": context_text})

    if not req.use_llm:
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
        if should_retrieve and db is not None:
            assistant_seq = max(len(req.history) - 1, 0) + 1
            try:
                db.add_documents(
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
                            db.delete(ids=ids_to_delete)
                    except Exception:
                        pass
            except Exception:
                pass

        yield _sse_event({"type": "done", "content": last_text})
    except Exception as e:
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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

