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


def _backend_reachable(backend_url: str) -> bool:
    try:
        # 根路径即可判断端口/服务是否可达。
        r = requests.get(backend_url, timeout=2)
        return r.status_code < 500
    except Exception:
        return False


def _submit_feedback(
    *,
    backend_url: str,
    session_id: str,
    question: str | None,
    answer: str,
    rating: str,
    reason: str | None,
    meta: Dict[str, Any] | None,
) -> tuple[bool, str]:
    payload = {
        "trace_id": (meta or {}).get("trace_id"),
        "session_id": session_id,
        "question": question,
        "answer": answer,
        "rating": rating,
        "reason": reason,
        "route_mode": (meta or {}).get("route_mode"),
        "match_score": (meta or {}).get("match_score"),
        "kb_max_score": (meta or {}).get("kb_max_score"),
    }
    try:
        r = requests.post(f"{backend_url}/feedback", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok", False):
            return False, data.get("error", "unknown feedback error")
        return True, ""
    except Exception as e:
        return False, str(e)


def _inject_ui_style() -> None:
    st.markdown(
        """
<style>
    .stApp {background: #f8fafc;}
    .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
    .stChatMessage {padding-top: 0.45rem; padding-bottom: 0.45rem;}
    .stCodeBlock {border-radius: 10px;}
    .app-header {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        background: white;
        margin-bottom: 0.75rem;
    }
    .app-subtitle {color: #475569; font-size: 0.92rem;}
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.2rem 0.6rem;
        font-size: 0.8rem;
        border: 1px solid #cbd5e1;
        margin-left: 0.45rem;
        color: #334155;
        background: #f1f5f9;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="智能客服", layout="wide")
    _inject_ui_style()

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
    if "feedback_state" not in st.session_state:
        st.session_state["feedback_state"] = {}

    default_persist = str(DEFAULT_PERSIST_DIR)
    backend_default = "http://127.0.0.1:8000"
    backend_status = _backend_reachable(backend_default)
    status_text = "后端已连接" if backend_status else "后端未连接"
    st.markdown(
        f"""
<div class="app-header">
  <div style="font-size:1.3rem;font-weight:700;">智能客服
    <span class="status-pill">{status_text}</span>
  </div>
  <div class="app-subtitle">基于课程知识库的检索增强问答，支持流式输出、来源追踪与降级生成。</div>
</div>
""",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("控制台")
        backend_url = st.text_input("后端地址（FastAPI）", value=backend_default)
        if _backend_reachable(backend_url):
            st.success("后端状态：已连接")
        else:
            st.error("后端状态：未连接")
        st.caption(f"Session ID: `{st.session_state['session_id'][:8]}`")
        with st.expander("检索参数", expanded=True):
            k = st.slider("检索段落数 k", min_value=1, max_value=10, value=3)
            use_llm = st.checkbox("调用 LLM 生成汇总回答", value=True)
            use_reranker = st.checkbox("启用 Cross-Encoder 重排序", value=True)
        with st.expander("模型与存储", expanded=False):
            persist_dir = st.text_input("向量库目录", value=default_persist)
            collection_name = st.text_input(
                "Collection 名称", value=DEFAULT_COLLECTION_NAME
            )
            embedding_model = st.text_input(
                "Embedding 模型名", value=DEFAULT_EMBEDDING_MODEL_NAME
            )
            llm_model = st.text_input("LLM 模型名（DashScope）", value="qwen-turbo")
        if st.button("清空会话", use_container_width=True):
            st.session_state["message"] = [
                {"role": "assistant", "content": "您好，有什么可以帮助你？"}
            ]
            st.session_state["memory"] = ConversationBufferMemory(
                return_messages=True,
            )
            st.rerun()

    for message in st.session_state["message"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("content"):
                msg_id = message.get("id")
                if msg_id:
                    fb = st.session_state["feedback_state"].get(msg_id, {})
                    submitted = fb.get("submitted", False)
                    if not submitted:
                        c1, c2 = st.columns([1, 1])
                        if c1.button("👍 有帮助", key=f"fb_up_{msg_id}"):
                            ok, err = _submit_feedback(
                                backend_url=backend_url,
                                session_id=st.session_state["session_id"],
                                question=message.get("question"),
                                answer=message.get("content", ""),
                                rating="up",
                                reason=None,
                                meta=message.get("meta"),
                            )
                            if ok:
                                st.session_state["feedback_state"][msg_id] = {
                                    "submitted": True,
                                    "rating": "up",
                                }
                                st.rerun()
                            else:
                                st.warning(f"反馈提交失败: {err}")

                        down_reason = c2.selectbox(
                            "👎 原因",
                            options=[
                                "答非所问",
                                "不准确",
                                "信息不足",
                                "太慢",
                                "其他",
                            ],
                            index=0,
                            key=f"fb_reason_{msg_id}",
                        )
                        if c2.button("提交差评", key=f"fb_down_{msg_id}"):
                            ok, err = _submit_feedback(
                                backend_url=backend_url,
                                session_id=st.session_state["session_id"],
                                question=message.get("question"),
                                answer=message.get("content", ""),
                                rating="down",
                                reason=down_reason,
                                meta=message.get("meta"),
                            )
                            if ok:
                                st.session_state["feedback_state"][msg_id] = {
                                    "submitted": True,
                                    "rating": "down",
                                    "reason": down_reason,
                                }
                                st.rerun()
                            else:
                                st.warning(f"反馈提交失败: {err}")
                    else:
                        st.caption("反馈已提交，感谢你的反馈。")

    prompt = st.chat_input("输入问题…")

    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        user_msg_id = str(uuid.uuid4())
        st.session_state["message"].append(
            {"id": user_msg_id, "role": "user", "content": prompt}
        )
        st.session_state["memory"].chat_memory.add_user_message(prompt)

        # 发送给后端的历史：包含已完成的上下文（不包含本轮的“正在生成”的回复）
        history_payload: List[Dict[str, str]] = list(st.session_state["message"])

        with st.chat_message("assistant"):
            with st.spinner("AI思考中..."):
                context_text = ""
                context_placeholder = st.empty()
                context_area = None

                answer_area = st.empty()
                current_answer = ""
                route_meta = None

                try:
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
                            # 降级为纯模型生成时，不展示“检索段落”栏位。
                            if event_content.strip() and event_content.strip() != "没有检索到所需内容":
                                with context_placeholder.container():
                                    with st.expander("检索到的相关段落", expanded=False):
                                        context_area = st.empty()
                                        context_area.markdown(event_content)
                            else:
                                context_placeholder.empty()
                        elif event_type == "answer":
                            current_answer = event_content
                            answer_area.markdown(current_answer)
                        elif event_type == "meta":
                            route_meta = event_content
                        elif event_type == "done":
                            if not current_answer and event_content:
                                current_answer = event_content
                        elif event_type == "error":
                            answer_area.markdown(f"错误：{event_content}")
                            current_answer = event_content
                except requests.RequestException as e:
                    current_answer = f"后端请求失败：{e}"
                    answer_area.markdown(current_answer)
                finally:
                    if route_meta:
                        route_mode = route_meta.get("route_mode", "-")
                        trace_id = route_meta.get("trace_id", "-")
                        match_score = route_meta.get("match_score")
                        kb_max_score = route_meta.get("kb_max_score")
                        topic = route_meta.get("topic", "-")
                        match_gate = route_meta.get("match_gate")
                        usable_threshold = route_meta.get("usable_threshold")
                        knowledge_version = route_meta.get("knowledge_version", "-")
                        st.caption(
                            "路由: `{route_mode}` | topic={topic} | match_score={match_score} | "
                            "match_gate={match_gate} | kb_max_score={kb_max_score} | "
                            "usable_threshold={usable_threshold} | kv={knowledge_version} | trace_id={trace_id}".format(
                                route_mode=route_mode,
                                topic=topic,
                                match_score=match_score,
                                match_gate=match_gate,
                                kb_max_score=kb_max_score,
                                usable_threshold=usable_threshold,
                                knowledge_version=knowledge_version,
                                trace_id=trace_id,
                            )
                        )
                        source_refs = route_meta.get("source_refs") or []
                        if source_refs:
                            with st.expander("来源验证", expanded=False):
                                for r in source_refs:
                                    label = r.get("label", "来源")
                                    hint = r.get("hint", "")
                                    url = r.get("url", "")
                                    if url:
                                        st.markdown(f"- [{label}]({url})  \n  {hint}")
                                    else:
                                        st.markdown(f"- {label}  \n  {hint}")

                        if route_meta.get("degraded"):
                            suggestions = route_meta.get("suggestions") or []
                            if suggestions:
                                st.info("当前为降级回答，建议这样提问以触发课件检索：")
                                for s in suggestions[:3]:
                                    st.markdown(f"- {s}")

        # 如果关闭了 LLM，只检索不生成：assistant 以检索内容作答（更适合简历演示）
        if not current_answer.strip() and use_llm is False:
            current_answer = context_text

        if not current_answer.strip():
            if not use_llm:
                current_answer = context_text or "未检索到相关段落。"
            else:
                current_answer = "未获取到模型输出（请检查后端/网络/密钥）。"

        st.session_state["message"].append(
            {
                "id": str(uuid.uuid4()),
                "role": "assistant",
                "content": current_answer,
                "question": prompt,
                "meta": route_meta,
            }
        )
        st.session_state["memory"].chat_memory.add_ai_message(current_answer)


if __name__ == "__main__":
    main()
