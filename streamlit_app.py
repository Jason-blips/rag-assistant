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
    :root {
        --bg-top: #c7d2fe;
        --bg-mid: #e0e7ff;
        --surface: #ffffff;
        --text: #0f172a;
        --muted: #475569;
        --brand: #4f46e5;
        --brand-dark: #3730a3;
        --brand-glow: #6366f1;
        --accent-violet: #7c3aed;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    .stApp {
        background: linear-gradient(165deg, var(--bg-top) 0%, var(--bg-mid) 28%, #f1f5f9 72%, #f8fafc 100%) !important;
        color: var(--text) !important;
    }
    .block-container {padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1000px;}
    /* 主对话槽：仅作用于包含底部输入框的那块卡片，避免污染消息内嵌的 bordered 容器 */
    main [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stChatInput"]) {
        background: rgba(255, 255, 255, 0.96) !important;
        border-radius: 20px !important;
        padding: 1rem 1.15rem 0.85rem !important;
        box-shadow: 0 12px 48px rgba(30, 27, 75, 0.12) !important;
        border: 1px solid rgba(129, 140, 248, 0.45) !important;
    }
    .stMarkdown, .stMarkdown p {color: var(--text) !important;}
    .stCodeBlock {border-radius: 10px; border: 1px solid #c7d2fe !important;}
    [data-testid="stChatMessage"] {
        border-radius: 14px !important;
        border: 1px solid #94a3b8 !important;
        background: #ffffff !important;
        padding: 0.55rem 1rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        border: none !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) li,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) span {
        color: #ffffff !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        border-left: 5px solid var(--brand) !important;
        border-color: #cbd5e1 !important;
        border-left-color: var(--brand) !important;
        background: #ffffff !important;
    }
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 2px solid var(--brand) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.18) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent-violet) !important;
        box-shadow: 0 0 0 4px rgba(124, 58, 237, 0.22), 0 8px 24px rgba(79, 70, 229, 0.2) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 0.98rem !important;
        color: var(--text) !important;
    }
    [data-testid="stChatInput"] button {
        background: linear-gradient(180deg, var(--brand-glow) 0%, var(--brand) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
    }
    .stTextInput input, .stSelectbox select {
        border-radius: 10px !important;
        border: 1px solid #94a3b8 !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--brand) !important;
        border-color: var(--brand-dark) !important;
    }
    .stCheckbox label p {color: var(--text) !important;}
    .stButton > button {
        border-radius: 10px !important;
        border: 2px solid var(--brand-dark) !important;
        background: linear-gradient(180deg, var(--brand-glow) 0%, var(--brand) 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
    }
    .stButton > button:hover {
        border-color: #312e81 !important;
        background: linear-gradient(180deg, #6366f1 0%, var(--brand-dark) 100%) !important;
        color: #ffffff !important;
        filter: brightness(1.05);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
    }
    .stButton > button:active {
        transform: translateY(2px);
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.35) !important;
    }
    .app-header {
        border-radius: 16px;
        padding: 1rem 1.15rem 1.1rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(125deg, #1e1b4b 0%, #312e81 42%, #5b21b6 100%);
        border: 1px solid #312e81;
        box-shadow: 0 12px 40px rgba(30, 27, 75, 0.45);
    }
    .app-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.03em;
        line-height: 1.25;
    }
    .app-subtitle {
        color: rgba(226, 232, 240, 0.92) !important;
        font-size: 0.9rem;
        margin-top: 0.35rem;
        line-height: 1.5;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 0.5rem;
        vertical-align: middle;
    }
    .app-header .status-pill.connected {
        color: #022c22;
        border: 1px solid #34d399;
        background: #6ee7b7;
    }
    .app-header .status-pill.disconnected {
        color: #450a0a;
        border: 1px solid #f87171;
        background: #fca5a5;
    }
    .meta-box {
        border: 2px solid #a5b4fc;
        background: linear-gradient(180deg, #eef2ff 0%, #e0e7ff 100%);
        border-radius: 12px;
        padding: 0.55rem 0.75rem;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
    }
    .chip {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 600;
        color: #312e81;
        background: #c7d2fe;
        border: 1px solid #818cf8;
        border-radius: 999px;
        padding: 0.15rem 0.55rem;
        margin: 0.12rem 0.28rem 0.12rem 0;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def _safe_text(v: Any) -> str:
    if v is None:
        return "-"
    s = str(v).strip()
    return s if s else "-"


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
    if "last_route_meta" not in st.session_state:
        st.session_state["last_route_meta"] = {}

    default_persist = str(DEFAULT_PERSIST_DIR)
    backend_default = "http://127.0.0.1:8000"
    backend_status = _backend_reachable(backend_default)
    status_text = "后端已连接" if backend_status else "后端未连接"
    status_class = "connected" if backend_status else "disconnected"
    st.markdown(
        f"""
<div class="app-header">
  <div class="app-title">智能客服
    <span class="status-pill {status_class}">{status_text}</span>
  </div>
  <div class="app-subtitle">基于课程知识库的检索增强问答，支持流式输出、来源追踪与降级生成。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    backend_url = backend_default
    persist_dir = default_persist
    collection_name = DEFAULT_COLLECTION_NAME
    embedding_model = DEFAULT_EMBEDDING_MODEL_NAME
    llm_model = "qwen-turbo"
    k = 3
    use_llm = True
    use_reranker = True
    show_debug = False

    _sp_left, _chat_main, _sp_right = st.columns([1, 4, 1])
    with _chat_main:
        with st.container(border=True):
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
                history_payload: List[Dict[str, str]] = list(
                    st.session_state["message"]
                )

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
                                    if (
                                        event_content.strip()
                                        and event_content.strip()
                                        != "没有检索到所需内容"
                                    ):
                                        with context_placeholder.container():
                                            with st.expander(
                                                "检索到的相关段落", expanded=False
                                            ):
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
                                st.session_state["last_route_meta"] = route_meta
                                route_mode = route_meta.get("route_mode", "-")
                                trace_id = route_meta.get("trace_id", "-")
                                match_score = route_meta.get("match_score")
                                kb_max_score = route_meta.get("kb_max_score")
                                topic = route_meta.get("topic", "-")
                                match_gate = route_meta.get("match_gate")
                                usable_threshold = route_meta.get("usable_threshold")
                                knowledge_version = route_meta.get(
                                    "knowledge_version", "-"
                                )
                                if show_debug:
                                    st.markdown(
                                        """
<div class="meta-box">
  <span class="chip">route: {route_mode}</span>
  <span class="chip">topic: {topic}</span>
  <span class="chip">match: {match_score}</span>
  <span class="chip">gate: {match_gate}</span>
  <span class="chip">kb_max: {kb_max_score}</span>
  <span class="chip">usable: {usable_threshold}</span>
  <span class="chip">kv: {knowledge_version}</span>
  <span class="chip">trace: {trace_id}</span>
</div>
""".format(
                                            route_mode=_safe_text(route_mode),
                                            topic=_safe_text(topic),
                                            match_score=_safe_text(match_score),
                                            match_gate=_safe_text(match_gate),
                                            kb_max_score=_safe_text(kb_max_score),
                                            usable_threshold=_safe_text(
                                                usable_threshold
                                            ),
                                            knowledge_version=_safe_text(
                                                knowledge_version
                                            ),
                                            trace_id=_safe_text(trace_id),
                                        ),
                                        unsafe_allow_html=True,
                                    )
                                source_refs = route_meta.get("source_refs") or []
                                if source_refs:
                                    with st.container(border=True):
                                        st.markdown("#### 来源验证")
                                        for r in source_refs:
                                            label = r.get("label", "来源")
                                            hint = r.get("hint", "")
                                            url = r.get("url", "")
                                            if url:
                                                st.markdown(
                                                    f"- [{label}]({url})  \n  {hint}"
                                                )
                                            else:
                                                st.markdown(f"- {label}  \n  {hint}")

                                if route_meta.get("degraded"):
                                    suggestions = route_meta.get("suggestions") or []
                                    if suggestions:
                                        with st.container(border=True):
                                            st.markdown("#### 当前为降级回答")
                                            st.caption("建议这样提问以触发课件检索：")
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
