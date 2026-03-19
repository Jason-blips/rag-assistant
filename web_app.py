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


def rag_answer(
    question: str,
    k: int,
    use_llm: bool,
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

    docs, scores = _retrieve(db, question, k=k, persist_dir=persist_dir)
    context_text = _format_docs_for_display(docs, scores)

    if not use_llm:
        return context_text, ""

    if not docs:
        return context_text, "未检索到相关段落，已中止 LLM 汇总。请尝试调整问题、增加 k 或先构建/更新向量库。"

    try:
        answer = _summarize_with_llm(question=question, docs=docs, model=llm_model)
    except Exception as e:
        answer = f"调用 LLM 失败：{e}"

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

