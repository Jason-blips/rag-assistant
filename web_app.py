import os
from typing import List, Tuple

import gradio as gr

from vectorstore_utils import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    load_vectorstore,
)
from ask_question import _retrieve, _summarize_with_llm, _doc_source_hint


def _load_db(
    persist_dir: str = str(DEFAULT_PERSIST_DIR),
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_NAME,
):
    return load_vectorstore(
        persist_directory=persist_dir,
        model_name=embedding_model,
        collection_name=collection_name,
    )


def _format_docs_for_display(docs, scores: List[float] | None) -> str:
    if not docs:
        return "（未检索到任何相关段落）"

    lines: list[str] = []
    for i, doc in enumerate(docs):
        score_str = ""
        if scores is not None and i < len(scores):
            score_str = f" | score={scores[i]:.4f}"
        hint = _doc_source_hint(doc)
        header = f"【段落 {i + 1}】{score_str}"
        if hint:
            header += f"\n来源：{hint}"
        lines.append(f"{header}\n{doc.page_content}")
    return "\n\n" + ("-" * 40 + "\n\n").join(lines)


def _build_source_suffix(docs, llm_model: str) -> str:
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


def rag_answer(
    question: str,
    k: int,
    use_llm: bool,
    use_reranker: bool,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    llm_model: str,
) -> Tuple[str, str]:
    if not question.strip():
        return "请输入问题。", ""

    try:
        db = _load_db(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
    except Exception as e:
        return f"加载向量库失败：{e}", ""

    docs, scores = _retrieve(
        db, question, k=k, persist_dir=persist_dir, use_reranker=use_reranker,
    )

    if not use_llm:
        # 先做“KB 降级判断”，避免把与课件无关的向量（例如对话向量）展示给用户。
        retrieval_usable_threshold = 0.50
        docs_for_llm = docs
        scores_for_display = scores
        if scores is not None and docs:
            kb_scores: list[float] = []
            for d, s in zip(docs, scores):
                md = getattr(d, "metadata", None) or {}
                if md.get("doc_type") == "kb":
                    kb_scores.append(s)
            # 双阈值逻辑（展示侧）：
            # 已检索（默认由 _retrieve 完成）但 KB 最高分达不到 0.5，视为检索结果不可用。
            if not kb_scores or max(kb_scores) < retrieval_usable_threshold:
                docs_for_llm = []
                scores_for_display = None
        context_text = _format_docs_for_display(docs_for_llm, scores_for_display)
        return context_text, ""

    try:
        # 条件降级：当检索“课件/KB 部分”的相似度整体偏低时，不把检索结果喂给 LLM。
        # 会话向量（doc_type=conversation）与当前问题高度相似，不能参与降级阈值判断。
        retrieval_usable_threshold = 0.50
        docs_for_llm = docs
        scores_for_display = scores
        if scores is not None and docs:
            kb_scores: list[float] = []
            for d, s in zip(docs, scores):
                md = getattr(d, "metadata", None) or {}
                if md.get("doc_type") == "kb":
                    kb_scores.append(s)
            # 双阈值逻辑（生成侧）：
            # 1) 触发检索阈值(0.3)由外部路由层负责；
            # 2) 若已检索但可用性达不到 0.5，则不把检索结果喂给 LLM。
            if not kb_scores or max(kb_scores) < retrieval_usable_threshold:
                docs_for_llm = []
                scores_for_display = None

        context_text = _format_docs_for_display(docs_for_llm, scores_for_display)
        answer = _summarize_with_llm(
            question=question,
            docs=docs_for_llm,
            model=llm_model,
        )
        source_suffix = _build_source_suffix(docs_for_llm, llm_model)
        answer = f"{answer}\n\n—— {source_suffix}"
    except Exception as e:
        answer = f"调用 LLM 失败：{e}"
        context_text = ""

    return context_text, answer


def main():
    default_persist = str(DEFAULT_PERSIST_DIR)

    with gr.Blocks(title="知识库智能问答助手") as demo:
        gr.Markdown(
            """
            ## University of York G400 Computer Science 课程知识库 - 智能问答助手

            - 支持从本地 PDF 课件构建知识库（先运行 `build_vector_db.py`）
            - 再在这里用自然语言提问，优先根据检索到的内容回答
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="问题",
                    placeholder="请输入你想问的问题，比如：UML 中类图的类代表什么？",
                    lines=3,
                )
                run_btn = gr.Button("开始检索并回答", variant="primary")

                with gr.Accordion("高级设置", open=False):
                    persist_dir = gr.Textbox(
                        label="向量库目录",
                        value=default_persist,
                        lines=1,
                    )
                    collection_name = gr.Textbox(
                        label="Collection 名称",
                        value=DEFAULT_COLLECTION_NAME,
                        lines=1,
                    )
                    embedding_model = gr.Textbox(
                        label="Embedding 模型名",
                        value=DEFAULT_EMBEDDING_MODEL_NAME,
                        lines=1,
                    )
                    llm_model = gr.Textbox(
                        label="LLM 模型名（DashScope）",
                        value="qwen-turbo",
                        lines=1,
                    )
                    k = gr.Slider(
                        label="检索段落数 k",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=3,
                    )
                    use_llm = gr.Checkbox(
                        label="是否调用 LLM 生成汇总回答",
                        value=True,
                    )
                    use_reranker = gr.Checkbox(
                        label="是否启用 Cross-Encoder 重排序",
                        value=True,
                    )

            with gr.Column(scale=5):
                context_box = gr.Textbox(
                    label="检索到的相关段落",
                    value="",
                    lines=18,
                )
                answer_box = gr.Textbox(
                    label="最终汇总回答（LLM）",
                    value="",
                    lines=10,
                )

        run_btn.click(
            rag_answer,
            inputs=[
                question,
                k,
                use_llm,
                use_reranker,
                persist_dir,
                collection_name,
                embedding_model,
                llm_model,
            ],
            outputs=[context_box, answer_box],
        )

    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))


if __name__ == "__main__":
    main()

