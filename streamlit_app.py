"""
对话式问答界面（Streamlit），与 web_app.py 共用同一套 RAG 逻辑。

必须用 Streamlit 进程启动（否则无 ScriptRunContext，session_state 不工作）：

    streamlit run streamlit_app.py

不要：python streamlit_app.py
"""
import json
import os
import time
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

# 首屏冷启动：一键示例问题（降低空窗与输入成本）
_STARTER_PROMPTS: List[str] = [
    "这门课主要讲什么？",
    "什么是检索增强生成（RAG）？",
    "知识库里的内容从哪来？",
]


def _welcome_messages() -> List[Dict[str, Any]]:
    return [{"role": "assistant", "content": "您好，有什么可以帮助你？"}]


def _memory_from_messages(messages: List[Dict[str, Any]]) -> ConversationBufferMemory:
    mem = ConversationBufferMemory(return_messages=True)
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            mem.chat_memory.add_user_message(content)
        elif role == "assistant":
            mem.chat_memory.add_ai_message(content)
    return mem


def _conv_title_from_messages(
    messages: List[Dict[str, Any]], fallback: str = "新对话"
) -> str:
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            t = (m.get("content") or "").strip().replace("\n", " ")
            return (t[:28] + "…") if len(t) > 28 else t
    return fallback


def _touch_active_conversation() -> None:
    cid = st.session_state.get("active_conv_id")
    convs = st.session_state.get("conversations")
    if not cid or not isinstance(convs, dict) or cid not in convs:
        return
    conv = convs[cid]
    conv["updated_at"] = time.time()
    conv["title"] = _conv_title_from_messages(conv["messages"])


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


def _backend_reachable(backend_url: str, *, timeout: float = 0.25) -> bool:
    base = (backend_url or "").rstrip("/")
    if not base:
        return False
    try:
        r = requests.get(f"{base}/health", timeout=timeout)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    try:
        r = requests.get(base, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def _topbar_badge_from_cache(backend_url: str, ttl_sec: float = 20.0) -> tuple[str, str]:
    """仅读缓存，不发起请求；无缓存时用于先画 UI 再延迟探测。"""
    now = time.monotonic()
    cache = st.session_state.get("_backend_ping")
    if (
        isinstance(cache, dict)
        and cache.get("url") == backend_url
        and (now - float(cache.get("t", 0))) < ttl_sec
    ):
        ok = bool(cache.get("ok"))
        return ("在线", "ok") if ok else ("离线", "bad")
    return ("检测中", "pending")


def _deferred_backend_ping(backend_url: str, ttl_sec: float = 20.0) -> None:
    """在页面主体渲染之后调用：缓存过期则探测一次并 rerun 刷新顶栏（避免阻塞首屏主区域）。"""
    now = time.monotonic()
    cache = st.session_state.get("_backend_ping")
    if (
        isinstance(cache, dict)
        and cache.get("url") == backend_url
        and (now - float(cache.get("t", 0))) < ttl_sec
    ):
        return
    ok = _backend_reachable(backend_url)
    st.session_state["_backend_ping"] = {"url": backend_url, "t": now, "ok": ok}
    st.rerun()


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
        --neo-bg: #0b0d12;
        /* 主面板与气泡用灰蓝系，避免纯白刺眼 */
        --neo-panel: #e8ecf2;
        --neo-ink: #0f172a;
        --neo-muted: #64748b;
        --neo-brand: #6366f1;
        --neo-brand-dim: #4f46e5;
        --neo-line: #c9d4e3;
        --neo-assistant-bg: #f0f3f9;
        --neo-user-bg-a: #e6eaf4;
        --neo-user-bg-b: #dce3f2;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent !important;}
    [data-testid="stToolbar"] {display: none;}
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        border-right: 1px solid rgba(148, 163, 184, 0.22) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] span,
    [data-testid="stSidebar"] [data-testid="stCaption"] {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    /* 侧栏主按钮不要用默认刺眼的红色 */
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="baseButton-primary"] {
        background: linear-gradient(180deg, #6366f1 0%, #4f46e5 100%) !important;
        border-color: #4338ca !important;
        color: #ffffff !important;
    }
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stDeployButton"] {display: none;}
    .stApp {
        background: radial-gradient(ellipse 90% 55% at 50% -18%, rgba(99, 102, 241, 0.16), transparent),
            linear-gradient(180deg, #334155 0%, #475569 42%, #64748b 100%) !important;
    }
    /* 勿在 .stApp 上写死深色字：聊天区若未套上浅色气泡会与深色底叠在一起看不见 */
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 5rem;
        max-width: 44rem !important;
        color: #0f172a;
        background: transparent !important;
    }
    /* 主内容区不要大白底，与页脚输入区视觉连贯 */
    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"],
    section.main {
        background: transparent !important;
    }
    /* 底部 Dock：与外圈渐变同色阶衔接 */
    [data-testid="stBottom"] {
        background: linear-gradient(180deg, rgba(100, 116, 139, 0) 0%, #64748b 38%, #787f8f 100%) !important;
        border-top: none !important;
        padding-top: 0.5rem !important;
    }
    .neo-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.65rem 1rem;
        margin-bottom: 0;
        background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-bottom: none;
        border-radius: 16px 16px 0 0;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .neo-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .neo-tagline {
        font-size: 0.78rem;
        color: #cbd5e1;
        margin-top: 0.2rem;
        line-height: 1.45;
        opacity: 0.95;
    }
    .neo-status {
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        white-space: nowrap;
    }
    .neo-status.ok { color: #6ee7b7; background: rgba(16, 185, 129, 0.12); border-color: rgba(52, 211, 153, 0.35); }
    .neo-status.bad { color: #fca5a5; background: rgba(248, 113, 113, 0.12); border-color: rgba(248, 113, 113, 0.35); }
    .neo-status.pending { color: #fde68a; background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.35); }
    /* 主对话面板：用 .block-container 限定，避免 section.main 在部分版本下匹配不到 */
    .block-container > div [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--neo-panel) !important;
        color: #0f172a !important;
        border-radius: 0 0 16px 16px !important;
        border: 1px solid rgba(100, 116, 139, 0.22) !important;
        border-top: 1px solid var(--neo-line) !important;
        padding: 0.75rem 0.85rem 1rem !important;
        box-shadow: 0 20px 40px -14px rgba(15, 23, 42, 0.35) !important;
    }
    .block-container > div [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        background: #dfe6f0 !important;
        color: #0f172a !important;
        border: 1px solid var(--neo-line) !important;
        box-shadow: none !important;
        padding: 0.45rem 0.6rem !important;
    }
    [data-testid="stChatMessage"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0.35rem 0 !important;
        margin-bottom: 0.35rem !important;
    }
    [data-testid="stChatMessage"] > div {
        gap: 0.65rem !important;
        align-items: flex-start !important;
    }
    /* 避免正文列 flex 撑满整行，否则气泡会被拉成一条 */
    [data-testid="stChatMessage"] > div > div:has([data-testid="stMarkdownContainer"]) {
        flex: 0 1 auto !important;
        max-width: 100% !important;
        min-width: 0 !important;
    }
    /* 头像：覆盖默认橙/红底 */
    [data-testid="chatAvatarIcon-assistant"],
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(145deg, #6366f1, #7c3aed) !important;
        color: #fff !important;
    }
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(145deg, #4f46e5, #6366f1) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stImage"] img {
        border-radius: 10px !important;
    }
    [data-testid="stChatMessage"] [data-baseweb="avatar"] {
        background: linear-gradient(145deg, #6366f1, #7c3aed) !important;
    }
    /* 每条消息正文：浅色气泡；inline-block + max-width 让短句随文字变窄，长文仍自动换行 */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        display: inline-block !important;
        vertical-align: top !important;
        width: auto !important;
        max-width: min(100%, 42rem) !important;
        box-sizing: border-box !important;
        background: var(--neo-assistant-bg) !important;
        border: 1px solid var(--neo-line) !important;
        border-radius: 14px !important;
        padding: 0.55rem 0.85rem !important;
        color: #0f172a !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
        color: #0f172a !important;
    }
    /* 支持 :has 时区分用户气泡 */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
        background: linear-gradient(135deg, var(--neo-user-bg-a) 0%, var(--neo-user-bg-b) 100%) !important;
        border: 1px solid #b8c5db !important;
    }
    [data-testid="stChatMessage"] label,
    [data-testid="stChatMessage"] [data-testid="stCaption"],
    [data-testid="stChatMessage"] .stMarkdown label {
        color: #334155 !important;
    }
    [data-testid="stChatMessage"] [data-baseweb="select"] > div {
        color: #0f172a !important;
    }
    .stCodeBlock { border-radius: 10px !important; border: 1px solid var(--neo-line) !important; }
    /* 底部输入：浮在深色底上 */
    [data-testid="stChatInput"] {
        background: #f0f3f9 !important;
        border: 1px solid rgba(100, 116, 139, 0.35) !important;
        border-radius: 999px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--neo-brand) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), 0 12px 40px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 0.95rem !important;
        color: var(--neo-ink) !important;
    }
    [data-testid="stChatInput"] button {
        background: var(--neo-brand) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 999px !important;
    }
    /* 消息内反馈：紧凑、偏「工具条」而非表单 */
    [data-testid="stChatMessage"] .stButton > button {
        background: #f0f3f9 !important;
        color: var(--neo-brand-dim) !important;
        border: 1px solid #c7d2fe !important;
        box-shadow: none !important;
        font-weight: 500 !important;
        min-height: 2.25rem !important;
        padding: 0.2rem 0.75rem !important;
        font-size: 0.8125rem !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] .stButton > button:hover {
        background: #eef2ff !important;
        border-color: var(--neo-brand) !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        border: 1px solid var(--neo-line) !important;
        border-radius: 10px !important;
        background: #e8edf5 !important;
        margin-top: 0.35rem !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary,
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary span {
        color: #475569 !important;
        font-size: 0.8125rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        padding-top: 0.25rem !important;
        gap: 0.35rem !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] .stSelectbox {
        margin-bottom: 0.35rem !important;
    }
    .stSelectbox label, .stSelectbox [data-baseweb="select"] {
        font-size: 0.85rem !important;
    }
    /* 首屏示例问题：secondary 按钮（与反馈区分） */
    [data-testid="stBaseButton-secondary"] {
        background: #e8edf5 !important;
        color: #334155 !important;
        border: 1px solid #e2e8f0 !important;
        font-weight: 500 !important;
        font-size: 0.8125rem !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        min-height: 2.35rem !important;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        background: #eef2ff !important;
        border-color: #c7d2fe !important;
        color: #312e81 !important;
    }
    .neo-prompt-hint {
        font-size: 0.75rem;
        font-weight: 600;
        color: #64748b;
        margin: 0.5rem 0 0.4rem 0;
        letter-spacing: 0.02em;
    }
    .meta-box {
        border: 1px solid #b8c5db;
        background: #e8edf5;
        border-radius: 12px;
        padding: 0.5rem 0.65rem;
        margin-top: 0.45rem;
    }
    .chip {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        color: #3730a3;
        background: #e0e7ff;
        border: 1px solid #a5b4fc;
        border-radius: 999px;
        padding: 0.12rem 0.45rem;
        margin: 0.08rem 0.2rem 0.08rem 0;
    }
    /* 生成中：spinner 默认在深色页上对比度差，在面板内加浅底+深色字 */
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] {
        background: #e2e9f3 !important;
        padding: 0.65rem 1rem !important;
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7) !important;
        margin: 0.35rem 0 0.5rem 0 !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] p,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] span,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] label {
        color: #0f172a !important;
        font-weight: 500 !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] .stSpinner {
        background: #e2e9f3 !important;
        padding: 0.65rem 1rem !important;
        border-radius: 12px !important;
        border: 1px solid #cbd5e1 !important;
        margin: 0.35rem 0 0.5rem 0 !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] .stSpinner p,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] .stSpinner span {
        color: #0f172a !important;
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


def _render_sidebar_chats() -> None:
    with st.sidebar:
        st.markdown("##### 历史对话")
        if st.button("➕ 新建对话", use_container_width=True, type="primary"):
            new_cid = str(uuid.uuid4())
            new_sid = str(uuid.uuid4())
            welcome = _welcome_messages()
            st.session_state["conversations"][new_cid] = {
                "title": "新对话",
                "messages": welcome,
                "session_id": new_sid,
                "updated_at": time.time(),
            }
            st.session_state["active_conv_id"] = new_cid
            st.session_state["_conv_switched"] = True
            st.session_state["_run_stream_for"] = None
            st.rerun()
        st.divider()
        items = sorted(
            st.session_state["conversations"].items(),
            key=lambda x: float(x[1].get("updated_at", 0)),
            reverse=True,
        )
        for cid, data in items:
            title = str(data.get("title") or "未命名")
            mark = "● " if cid == st.session_state["active_conv_id"] else ""
            if st.button(
                f"{mark}{title}",
                key=f"sidebar_conv_{cid}",
                use_container_width=True,
            ):
                if cid != st.session_state["active_conv_id"]:
                    st.session_state["active_conv_id"] = cid
                    st.session_state["_conv_switched"] = True
                    st.session_state["_run_stream_for"] = None
                    st.rerun()
        st.caption("会话保存在本页；关闭或强刷标签页后会清空。")


def main():
    st.set_page_config(
        page_title="课程助手",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # 每次 run 都注入：Streamlit rerun 会重建 DOM，只注入一次会导致样式块丢失后整页退回默认浅色
    _inject_ui_style()

    # ---------- 多会话（侧边栏）：兼容旧版仅有 message / session_id ----------
    if "conversations" not in st.session_state:
        if "message" in st.session_state and isinstance(
            st.session_state["message"], list
        ):
            _cid = str(uuid.uuid4())
            _sid = st.session_state.get("session_id") or str(uuid.uuid4())
            _msgs = st.session_state["message"]
            st.session_state["conversations"] = {
                _cid: {
                    "title": _conv_title_from_messages(_msgs),
                    "messages": _msgs,
                    "session_id": _sid,
                    "updated_at": time.time(),
                }
            }
            st.session_state["active_conv_id"] = _cid
            st.session_state["session_id"] = _sid
        else:
            _cid = str(uuid.uuid4())
            _sid = str(uuid.uuid4())
            _msgs = _welcome_messages()
            st.session_state["conversations"] = {
                _cid: {
                    "title": "新对话",
                    "messages": _msgs,
                    "session_id": _sid,
                    "updated_at": time.time(),
                }
            }
            st.session_state["active_conv_id"] = _cid
            st.session_state["session_id"] = _sid
            st.session_state["message"] = _msgs

    _ac = st.session_state.get("active_conv_id")
    if _ac not in st.session_state["conversations"]:
        _ac = next(iter(st.session_state["conversations"]))
        st.session_state["active_conv_id"] = _ac

    _render_sidebar_chats()

    _conv = st.session_state["conversations"][st.session_state["active_conv_id"]]
    st.session_state["message"] = _conv["messages"]
    st.session_state["session_id"] = _conv["session_id"]

    if st.session_state.pop("_conv_switched", False):
        st.session_state["memory"] = _memory_from_messages(
            st.session_state["message"]
        )

    if "memory" not in st.session_state:
        st.session_state["memory"] = _memory_from_messages(
            st.session_state["message"]
        )
    if "feedback_state" not in st.session_state:
        st.session_state["feedback_state"] = {}
    if "last_route_meta" not in st.session_state:
        st.session_state["last_route_meta"] = {}
    if "_run_stream_for" not in st.session_state:
        st.session_state["_run_stream_for"] = None

    default_persist = str(DEFAULT_PERSIST_DIR)
    backend_default = os.getenv(
        "RAG_BACKEND_URL", "http://127.0.0.1:8000"
    ).rstrip("/")
    status_text, status_class = _topbar_badge_from_cache(backend_default)
    st.markdown(
        f"""
<div class="neo-topbar">
  <div>
    <div class="neo-brand">课程助手</div>
    <div class="neo-tagline">课程知识库 · 流式回答 · 来源可追溯</div>
  </div>
  <span class="neo-status {status_class}">{status_text}</span>
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

    with st.container(border=True):
            # 仅最新一条助手消息展示反馈控件，避免长会话时控件数量爆炸拖慢渲染
            last_feedback_msg_id: str | None = None
            for m in reversed(st.session_state["message"]):
                if m.get("role") == "assistant" and m.get("id"):
                    last_feedback_msg_id = str(m["id"])
                    break

            for message in st.session_state["message"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant" and message.get("content"):
                        msg_id = message.get("id")
                        if msg_id and str(msg_id) == last_feedback_msg_id:
                            fb = st.session_state["feedback_state"].get(msg_id, {})
                            submitted = fb.get("submitted", False)
                            if not submitted:
                                with st.expander(
                                    "对这条回答反馈（可选）", expanded=False
                                ):
                                    if st.button("👍 有帮助", key=f"fb_up_{msg_id}"):
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
                                            st.session_state["feedback_state"][
                                                msg_id
                                            ] = {
                                                "submitted": True,
                                                "rating": "up",
                                            }
                                            st.rerun()
                                        else:
                                            st.warning(f"反馈提交失败: {err}")

                                    st.caption("若不满意，请选择原因后提交")
                                    cr, cs = st.columns([5, 2])
                                    down_reason = cr.selectbox(
                                        "原因",
                                        options=[
                                            "答非所问",
                                            "不准确",
                                            "信息不足",
                                            "太慢",
                                            "其他",
                                        ],
                                        index=0,
                                        key=f"fb_reason_{msg_id}",
                                        label_visibility="collapsed",
                                    )
                                    if cs.button("提交差评", key=f"fb_down_{msg_id}"):
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
                                            st.session_state["feedback_state"][
                                                msg_id
                                            ] = {
                                                "submitted": True,
                                                "rating": "down",
                                                "reason": down_reason,
                                            }
                                            st.rerun()
                                        else:
                                            st.warning(f"反馈提交失败: {err}")
                            else:
                                st.caption("反馈已提交，感谢你的反馈。")
                        elif msg_id:
                            _ofb = st.session_state["feedback_state"].get(
                                msg_id, {}
                            )
                            if _ofb.get("submitted"):
                                st.caption("反馈已提交，感谢你的反馈。")

            if (
                len(st.session_state["message"]) == 1
                and st.session_state["message"][0].get("role") == "assistant"
                and not st.session_state.get("_run_stream_for")
            ):
                st.markdown(
                    '<p class="neo-prompt-hint">试试这样问</p>',
                    unsafe_allow_html=True,
                )
                _aid = st.session_state["active_conv_id"]
                sc = st.columns(len(_STARTER_PROMPTS))
                for i, q in enumerate(_STARTER_PROMPTS):
                    if sc[i].button(
                        q,
                        key=f"starter_{_aid}_{i}",
                        use_container_width=True,
                        type="secondary",
                    ):
                        st.session_state["message"].append(
                            {
                                "id": str(uuid.uuid4()),
                                "role": "user",
                                "content": q,
                            }
                        )
                        st.session_state["memory"].chat_memory.add_user_message(q)
                        st.session_state["_run_stream_for"] = q
                        _touch_active_conversation()
                        st.rerun()

            pending = st.session_state.get("_run_stream_for")
            if pending:
                stream_prompt = pending
                history_payload: List[Dict[str, str]] = list(
                    st.session_state["message"]
                )
                # 生成过程放在聊天气泡外：仅 st.spinner + 正文，完成后写入历史再以气泡展示
                thinking_slot = st.empty()
                context_placeholder = st.empty()
                answer_area = st.empty()
                context_text = ""
                current_answer = ""
                route_meta = None
                spinner_cleared = False

                def _end_thinking() -> None:
                    nonlocal spinner_cleared
                    if not spinner_cleared:
                        thinking_slot.empty()
                        spinner_cleared = True

                try:
                    with thinking_slot:
                        with st.spinner("AI 正在思考中"):
                            for event in _chat_stream_request(
                                backend_url=backend_url,
                                session_id=st.session_state["session_id"],
                                prompt=stream_prompt,
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
                                    if (
                                        event_content.strip()
                                        and event_content.strip()
                                        != "没有检索到所需内容"
                                    ):
                                        _end_thinking()
                                        with context_placeholder.container():
                                            with st.expander(
                                                "相关段落", expanded=False
                                            ):
                                                context_area = st.empty()
                                                context_area.markdown(
                                                    event_content
                                                )
                                    else:
                                        context_placeholder.empty()
                                elif event_type == "answer":
                                    _end_thinking()
                                    current_answer = event_content
                                    answer_area.markdown(current_answer)
                                elif event_type == "meta":
                                    route_meta = event_content
                                elif event_type == "done":
                                    if not current_answer and event_content:
                                        current_answer = event_content
                                        _end_thinking()
                                        answer_area.markdown(current_answer)
                                elif event_type == "error":
                                    _end_thinking()
                                    answer_area.markdown(
                                        f"错误：{event_content}"
                                    )
                                    current_answer = event_content
                except requests.RequestException as e:
                    _end_thinking()
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
                        "question": stream_prompt,
                        "meta": route_meta,
                    }
                )
                st.session_state["memory"].chat_memory.add_ai_message(
                    current_answer
                )
                _touch_active_conversation()
                st.session_state["_run_stream_for"] = None
                st.rerun()

    prompt = st.chat_input("输入问题…")
    if prompt:
        user_msg_id = str(uuid.uuid4())
        st.session_state["message"].append(
            {"id": user_msg_id, "role": "user", "content": prompt}
        )
        st.session_state["memory"].chat_memory.add_user_message(prompt)
        st.session_state["_run_stream_for"] = prompt
        _touch_active_conversation()
        st.rerun()

    _deferred_backend_ping(backend_default)


# streamlit run 会执行本文件且 __name__ 为 __main__；部分版本下 get_script_run_ctx()
# 在首次执行时仍为 None，若据此 sys.exit 会导致页面空白「无法加载」。因此始终调用 main()。
if __name__ == "__main__":
    main()
