"""
对话式问答界面（Streamlit），与 web_app.py 共用同一套 RAG 逻辑。
运行：streamlit run streamlit_app.py
"""
import json
import uuid
from typing import Any, Dict, Iterator, List

import requests
import streamlit as st

from langchain_classic.memory.buffer import ConversationBufferMemory

from vectorstore_utils import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
)


def _parse_sse_lines(resp: requests.Response) -> Iterator[Dict[str, Any]]:
    """
    从 SSE 响应流中解析形如：`data: {...}` 的 JSON 事件。
    """
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith("data:"):
            payload_str = raw_line[len("data:") :].strip()
            if payload_str:
                yield json.loads(payload_str)


def _chat_stream_request(
    *,
    backend_url: str,
    session_id: str,
    prompt: str,
    history: List[Dict[str, str]],
    k: int,
    use_llm: bool,
    use_reranker: bool,
    persist_dir: str,
    collection_name: str,
    embedding_model: str,
    llm_model: str,
) -> Iterator[Dict[str, Any]]:
    req_payload = {
        "question": prompt,
        "session_id": session_id,
        "history": history,
        "k": k,
        "use_llm": use_llm,
        "use_reranker": use_reranker,
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
    }

    resp = requests.post(
        f"{backend_url}/chat/stream",
        json=req_payload,
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    return _parse_sse_lines(resp)


def main():
    st.set_page_config(page_title="智能客服", layout="wide")
    st.title("智能客服")
    st.divider()

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())

    if "message" not in st.session_state:
        st.session_state["message"] = [
            {"role": "assistant", "content": "您好，有什么可以帮助你？"}
        ]

    if "memory" not in st.session_state:
        # 用 LangChain 的 ConversationBufferMemory 保存历史对话（用于后续扩展/一致性）。
        # 注意：向量检索仍然以 st.session_state["message"] 为准。
        st.session_state["memory"] = ConversationBufferMemory(
            return_messages=True,
        )

    default_persist = str(DEFAULT_PERSIST_DIR)

    with st.sidebar:
        st.subheader("检索与模型")
        backend_url = st.text_input("后端地址（FastAPI）", value="http://127.0.0.1:8000")
        persist_dir = st.text_input("向量库目录", value=default_persist)
        collection_name = st.text_input(
            "Collection 名称", value=DEFAULT_COLLECTION_NAME
        )
        embedding_model = st.text_input(
            "Embedding 模型名", value=DEFAULT_EMBEDDING_MODEL_NAME
        )
        llm_model = st.text_input("LLM 模型名（DashScope）", value="qwen-turbo")
        k = st.slider("检索段落数 k", min_value=1, max_value=10, value=3)
        use_llm = st.checkbox("调用 LLM 生成汇总回答", value=True)
        use_reranker = st.checkbox("启用 Cross-Encoder 重排序", value=True)

    for message in st.session_state["message"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("输入问题…")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["message"].append({"role": "user", "content": prompt})
        st.session_state["memory"].chat_memory.add_user_message(prompt)

        # 发送给后端的历史：包含已完成的上下文（不包含本轮的“正在生成”的回复）
        history_payload: List[Dict[str, str]] = list(st.session_state["message"])

        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                with st.expander("检索到的相关段落", expanded=False):
                    context_area = st.empty()
                    context_area.markdown("")
                context_text = ""

                answer_area = st.empty()
                current_answer = ""

                for event in _chat_stream_request(
                    backend_url=backend_url,
                    session_id=st.session_state["session_id"],
                    prompt=prompt,
                    history=history_payload,
                    k=k,
                    use_llm=use_llm,
                    use_reranker=use_reranker,
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                ):
                    event_type = event.get("type")
                    event_content = event.get("content") or ""

                    if event_type == "context":
                        context_text = event_content
                        context_area.markdown(event_content)
                    elif event_type == "answer":
                        current_answer = event_content
                        answer_area.markdown(current_answer)
                    elif event_type == "done":
                        if not current_answer and event_content:
                            current_answer = event_content
                    elif event_type == "error":
                        answer_area.markdown(f"错误：{event_content}")
                        current_answer = event_content

        # 如果关闭了 LLM，只检索不生成：assistant 以检索内容作答（更适合简历演示）
        if not current_answer.strip() and use_llm is False:
            current_answer = context_text

        if not current_answer.strip():
            if not use_llm:
                current_answer = context_text or "未检索到相关段落。"
            else:
                current_answer = "未获取到模型输出（请检查后端/网络/密钥）。"

        st.session_state["message"].append(
            {"role": "assistant", "content": current_answer}
        )
        st.session_state["memory"].chat_memory.add_ai_message(current_answer)


if __name__ == "__main__":
    main()
