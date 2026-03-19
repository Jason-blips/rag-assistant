import argparse
from collections.abc import Iterator
from typing import Any

from vectorstore_utils import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    infer_topic,
    hybrid_retrieve,
    load_vectorstore,
)


def _ensure_utf8_stdout() -> None:
    """
    Windows 控制台常见默认编码是 GBK，遇到 PDF 中的特殊字符（如 •）会导致打印报错。
    这里尽量把 stdout 切到 UTF-8；不支持时也不影响后续逻辑。
    """
    try:
        import sys

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))


def _format_context(docs) -> str:
    if not docs:
        return "没有检索到所需内容"
    parts: list[str] = []
    for i, doc in enumerate(docs):
        md = getattr(doc, "metadata", None) or {}
        if md.get("doc_type") == "conversation":
            role = md.get("role") or "对话"
            parts.append(f"[对话片段 {i + 1} | {role}]\n{doc.page_content}")
        else:
            parts.append(f"[段落 {i + 1}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _doc_source_hint(doc) -> str:
    md = getattr(doc, "metadata", None) or {}
    source = md.get("source") or md.get("file_path") or ""
    page = md.get("page")
    if source and page is not None:
        return f"{source}#page={page}"
    if source:
        return str(source)
    if page is not None:
        return f"page={page}"
    return ""


def _retrieve(
    db,
    question: str,
    k: int,
    persist_dir: str = str(DEFAULT_PERSIST_DIR),
    use_reranker: bool = True,
):
    """
    优先使用 BM25+向量混合检索(RRF 融合 + 可选 Reranker 精排)；
    如果 BM25 语料不可用则退化到纯向量检索。
    返回 (docs, scores?)，scores 若不可用则为 None。
    """
    topic_preference = infer_topic(question)
    try:
        return hybrid_retrieve(
            db=db, query=question, k=k,
            persist_directory=persist_dir,
            use_reranker=use_reranker,
            topic_preference=topic_preference,
        )
    except FileNotFoundError:
        pass

    try:
        pairs = db.similarity_search_with_relevance_scores(question, k=k)
        docs = [d for d, _ in pairs]
        scores = [s for _, s in pairs]
        if topic_preference:
            matched: list[tuple[Any, float]] = []
            for d, s in zip(docs, scores):
                doc_topic = (getattr(d, "metadata", None) or {}).get("topic")
                if doc_topic == topic_preference:
                    matched.append((d, s))
            if matched:
                docs = [d for d, _ in matched][:k]
                scores = [s for _, s in matched][:k]
        return docs, scores
    except Exception:
        retriever = db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)
        return docs, None


def _extract_dashscope_text(response) -> str:
    """
    DashScope SDK 不同版本返回结构可能略有差异，这里做尽量稳健的提取。
    """
    try:
        choices = response.output.choices
        if not choices:
            return ""
        msg = choices[0].message
        content = getattr(msg, "content", None)
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        if isinstance(content, str):
            return content
    except Exception:
        pass

    try:
        return str(response.output)
    except Exception:
        return ""


def _summarize_with_llm(question: str, docs, model: str) -> str:
    import os
    from http import HTTPStatus

    import dashscope

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，请确认已写入系统变量。")

    dashscope.api_key = api_key

    context_text = _format_context(docs)
    system_prompt = (
        "你是一个严谨的Computer Science专业学习助教。请优先依据给定资料回答问题；"
        "如果资料不足，请明确说明“不足以确定”，不要编造。"
        "题目本身是英文，请同时给出对应的英文标准答案。"
    )
    user_prompt = f"""下面是与问题相关的资料片段（可能不完整）：
{context_text}

问题：
{question}

请严格按下面格式输出（不要添加额外小标题）：
1) 中文最终回答：一段简洁的中文回答。 + 英文标准答案：一段准确的英文作答，适合作为英文考试/作业的标准答案。
2) 关键依据：引用你使用到的段落编号，例如“段落 1、3”，必要时可简单说明理由。
"""

    resp = dashscope.Generation.call(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        result_format="message",
    )

    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"调用 DashScope 失败：{getattr(resp, 'message', resp)}")

    text = _extract_dashscope_text(resp).strip()
    if not text:
        raise RuntimeError("LLM 返回为空，请稍后重试或检查 DashScope SDK/模型配置。")
    return text


def _format_history_for_prompt(history: list[dict[str, Any]]) -> str:
    """
    将前端传入的历史消息格式化为提示词片段（用于保证对话连贯性）。
    """
    if not history:
        return ""

    parts: list[str] = []
    # 只取最近若干轮，避免 prompt 过长
    for msg in history[-10:]:
        role = (msg.get("role") or "").lower()
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        if role == "user":
            prefix = "用户"
        elif role in {"assistant", "bot"}:
            prefix = "助手"
        else:
            prefix = role or "对话"

        parts.append(f"{prefix}：{content}")

    return "\n".join(parts)


def _summarize_with_llm_stream(
    question: str,
    docs,
    model: str,
    history: list[dict[str, Any]] | None = None,
) -> Iterator[str]:
    """
    DashScope 流式生成：每次 yield 当前累计的回答文本。
    """
    import os
    from http import HTTPStatus

    import dashscope

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，请确认已写入系统变量。")

    dashscope.api_key = api_key

    context_text = _format_context(docs)
    history_text = _format_history_for_prompt(history or [])

    system_prompt = (
        "你是一个严谨的Computer Science专业学习助教。请优先依据给定资料回答问题；"
        "如果资料不足，请明确说明“不足以确定”，不要编造。"
        "题目本身是英文，请同时给出对应的英文标准答案。"
    )
    user_prompt = f"""对话历史（用于保持连贯性，必要时参考，不要替代资料）：
{history_text or "（无）"}

下面是与问题相关的资料片段（可能不完整）：
{context_text}

问题：
{question}

请严格按下面格式输出（不要添加额外小标题）：
1) 中文最终回答：一段简洁的中文回答。 + 英文标准答案：一段准确的英文作答，适合作为英文考试/作业的标准答案。
2) 关键依据：引用你使用到的段落编号，例如“段落 1、3”，必要时可简单说明理由。
"""

    resp_stream = dashscope.Generation.call(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        result_format="message",
        stream=True,
        # 对于常见 qwen 模型，显式设为 False 触发 SDK 将增量内容合并为“累计全文”，
        # 前端就可以直接替换显示而不是做字符级 diff。
        incremental_output=False,
    )

    # 迭代每个 chunk；由于上面设定了合并策略，这里抽取到的是“当前累计文本”
    for chunk in resp_stream:
        if getattr(chunk, "status_code", None) and chunk.status_code != HTTPStatus.OK:
            raise RuntimeError(f"调用 DashScope 失败：{getattr(chunk, 'message', chunk)}")
        text = _extract_dashscope_text(chunk).strip()
        if text:
            yield text


def ask(
    question: str,
    k: int = 3,
    persist_dir: str = str(DEFAULT_PERSIST_DIR),
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL_NAME,
    use_llm: bool = True,
    llm_model: str = "qwen-turbo",
    print_context: bool = True,
    use_reranker: bool = True,
) -> None:
    """
    先从向量库检索相关段落（混合检索 + Reranker），再（可选）调用阿里云 LLM 生成最终汇总答案。
    """
    db = load_vectorstore(
        persist_directory=persist_dir,
        model_name=embedding_model,
        collection_name=collection_name,
    )
    docs, scores = _retrieve(
        db=db, question=question, k=k,
        persist_dir=persist_dir, use_reranker=use_reranker,
    )

    print("=" * 50)
    print("你的问题：", question)
    print("=" * 50)

    if print_context:
        print(f"将会尝试从 {persist_dir} 找到答案：\n")
        if not docs:
            print("（未检索到任何段落）")
        for i, doc in enumerate(docs):
            hint = _doc_source_hint(doc)
            score_text = ""
            if scores is not None and i < len(scores):
                score_text = f" | score={scores[i]:.4f}"
            print(f"【找到的段落 {i + 1}】{score_text}")
            if hint:
                print(f"来源：{hint}")
            _safe_print(doc.page_content)
            print("=" * 50)

    if use_llm:
        if not docs:
            raise RuntimeError("未检索到相关段落，已中止 LLM 汇总（请尝试调整问题/增加 k 或重建向量库）。")
        answer = _summarize_with_llm(question=question, docs=docs, model=llm_model)
        print("\n" + "=" * 50)
        print("📌 最终汇总答案（LLM 基于检索资料生成）")
        print("=" * 50)
        print(answer)


def main() -> None:
    """
    命令行入口：
    - --question / -q：问题
    - --k：返回段落数
    - --persist-dir：向量库目录
    - --no-llm：只检索不调用 LLM
    - --llm-model：DashScope 模型名（默认 qwen-turbo）
    - --no-context：不打印检索段落，只输出最终答案
    - （无参数运行）：进入交互式提问模式
    """
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="RAG 检索 +（可选）LLM 汇总回答")
    parser.add_argument("--question", "-q", type=str, default=None, help="要提问的问题内容")
    parser.add_argument("--k", type=int, default=3, help="检索返回的段落数量（默认: 3）")
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=str(DEFAULT_PERSIST_DIR),
        help="向量库持久化目录（默认: ./vector_db）",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection 名称（默认: {DEFAULT_COLLECTION_NAME}）",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_NAME,
        help=f"Embedding 模型名（默认: {DEFAULT_EMBEDDING_MODEL_NAME}）",
    )
    parser.add_argument("--no-llm", action="store_true", help="只检索不调用 LLM（默认会调用）")
    parser.add_argument("--llm-model", type=str, default="qwen-turbo", help="DashScope 模型名（默认: qwen-turbo）")
    parser.add_argument("--no-context", action="store_true", help="不打印检索段落（默认会打印）")
    parser.add_argument("--no-reranker", action="store_true", help="跳过 Cross-Encoder 重排序（默认启用）")

    args = parser.parse_args()

    if args.question:
        ask(
            question=args.question,
            k=args.k,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            use_llm=not args.no_llm,
            llm_model=args.llm_model,
            print_context=not args.no_context,
            use_reranker=not args.no_reranker,
        )
        return

    # 交互式提问：不传 --question 时启用
    print("进入交互模式：输入问题后回车开始检索与总结；输入 exit 退出。\n")
    while True:
        try:
            q = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            return

        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            print("退出。")
            return

        ask(
            question=q,
            k=args.k,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
            use_llm=not args.no_llm,
            llm_model=args.llm_model,
            print_context=not args.no_context,
            use_reranker=not args.no_reranker,
        )


if __name__ == "__main__":
    main()
