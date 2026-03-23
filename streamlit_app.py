"""
对话式问答界面（Streamlit），与 web_app.py 共用同一套 RAG 逻辑。

必须用 Streamlit 进程启动（否则无 ScriptRunContext，session_state 不工作）：

    streamlit run streamlit_app.py

不要：python streamlit_app.py
"""
import json
import os
import re
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


def _ensure_session_numbers(convs: Dict[str, Any]) -> None:
    """每条会话固定编号 session_no，侧栏显示「会话 N」；旧数据无编号时按 updated_at 升序补齐。"""
    if not convs:
        return
    max_sn = 0
    for c in convs.values():
        sn = c.get("session_no")
        if isinstance(sn, int):
            max_sn = max(max_sn, sn)
    ordered = sorted(
        convs.items(),
        key=lambda x: float(x[1].get("updated_at", 0)),
    )
    for _cid, c in ordered:
        if not isinstance(c.get("session_no"), int):
            max_sn += 1
            c["session_no"] = max_sn
    for c in convs.values():
        sn = c.get("session_no")
        if isinstance(sn, int):
            c["title"] = f"会话 {sn}"


def _next_session_no(convs: Dict[str, Any]) -> int:
    _ensure_session_numbers(convs)
    mx = 0
    for c in convs.values():
        sn = c.get("session_no")
        if isinstance(sn, int):
            mx = max(mx, sn)
    return mx + 1


def _touch_active_conversation() -> None:
    cid = st.session_state.get("active_conv_id")
    convs = st.session_state.get("conversations")
    if not cid or not isinstance(convs, dict) or cid not in convs:
        return
    conv = convs[cid]
    conv["updated_at"] = time.time()


def _api_chat_history(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """只传 role/content 给后端，避免多余字段；空内容跳过。"""
    out: List[Dict[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content or role not in ("user", "assistant"):
            continue
        out.append({"role": str(role), "content": content})
    return out


@st.cache_resource
def _http_session() -> requests.Session:
    """复用连接池：健康检查 / 流式问答 / 反馈 共用，减轻反复 TCP 握手。"""
    s = requests.Session()
    return s


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


def _chat_stream_post(
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
) -> requests.Response:
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
    resp = _http_session().post(
        f"{backend_url}/chat/stream",
        json=req_payload,
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    return resp


# 每次 rerun 只消费一批 SSE，才能渲染「停止生成」并关闭连接
_STREAM_BATCH_SIZE = 12


def _clear_stream_ui_state() -> None:
    """关闭流式 HTTP 连接并清理 session_state 中的迭代器（切换会话/停止时调用）。"""
    r = st.session_state.pop("_stream_resp", None)
    if r is not None:
        try:
            r.close()
        except Exception:
            pass
    st.session_state.pop("_stream_event_iter", None)
    st.session_state.pop("_stream_acc", None)


def _backend_reachable(backend_url: str, *, timeout: float = 0.25) -> bool:
    base = (backend_url or "").rstrip("/")
    if not base:
        return False
    sess = _http_session()
    try:
        r = sess.get(f"{base}/health", timeout=timeout)
        if r.status_code == 200:
            return True
    except Exception:
        pass
    try:
        r = sess.get(base, timeout=timeout)
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
        r = _http_session().post(f"{backend_url}/feedback", json=payload, timeout=10)
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
        /* 与 .block-container 一致，主栏与底部输入同一视觉宽度 */
        --neo-content-max: 44rem;
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
    /* 勿隐藏整条 stToolbar：新版 Streamlit 的「展开/收起侧栏」按钮在工具栏内，隐藏后侧栏被收起就再也打不开 */
    [data-testid="stToolbar"] {visibility: visible !important; display: flex !important;}
    /* 侧栏：与主区同系深灰，避免蓝紫渐变与亮主色 */
    [data-testid="stSidebar"] > div:first-child {
        background: #1a1a1a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4, [data-testid="stSidebar"] h5, [data-testid="stSidebar"] h6 {
        color: #a3a3a3 !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdown"] span,
    [data-testid="stSidebar"] [data-testid="stCaption"] {
        color: #a3a3a3 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
    }
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="baseButton-primary"] {
        background: #2a2a2a !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e5e5e5 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover,
    [data-testid="stSidebar"] [data-testid="baseButton-primary"]:hover {
        background: #333333 !important;
        border-color: rgba(255, 255, 255, 0.14) !important;
        color: #f5f5f5 !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #d4d4d4 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover,
    [data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
        color: #f5f5f5 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        background: rgba(0, 0, 0, 0.15) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary span {
        color: #a3a3a3 !important;
    }
    [data-testid="stDecoration"] {display: none;}
    [data-testid="stDeployButton"] {display: none;}
    .stApp {
        /* ChatGPT 式：中性深灰底，减少大面积紫渐变 */
        background: #212121 !important;
    }
    /* 勿在 .stApp 上写死深色字：聊天区若未套上浅色气泡会与深色底叠在一起看不见 */
    /* chat_input 固定在底部（stBottom），主区需多留底边距，否则最后几条消息像被「压住」 */
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 7.5rem;
        max-width: var(--neo-content-max) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        color: #e2e8f0;
        background: transparent !important;
    }
    section.main {
        scroll-padding-bottom: 6rem;
    }
    /* 主内容区不要大白底，与页脚输入区视觉连贯 */
    [data-testid="stAppViewContainer"],
    [data-testid="stMainBlockContainer"],
    section.main {
        background: transparent !important;
    }
    /* 底部 Dock：与背景融在一起，减轻「整块盖在内容上」的悬浮感 */
    [data-testid="stBottom"] {
        background: linear-gradient(180deg, rgba(33, 33, 33, 0) 0%, rgba(33, 33, 33, 0.96) 38%, #212121 100%) !important;
        border-top: none !important;
        padding-top: 0.5rem !important;
        padding-bottom: env(safe-area-inset-bottom, 0) !important;
        box-shadow: none !important;
    }
    /* 底部输入与主对话区同宽居中，避免「窄聊宽框」割裂感 */
    [data-testid="stBottom"] > div {
        max-width: var(--neo-content-max) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        box-sizing: border-box !important;
    }
    /* 部分版本里输入框在 VerticalBlock 内：只限宽居中，避免与外层 padding 叠加 */
    [data-testid="stBottom"] [data-testid="stVerticalBlock"] {
        max-width: var(--neo-content-max) !important;
        margin-left: auto !important;
        margin-right: auto !important;
        box-sizing: border-box !important;
    }
    .neo-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.65rem 1rem;
        margin-bottom: 0;
        background: #2b2b2b !important;
        border: none !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }
    .neo-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .neo-tagline {
        font-size: 0.8125rem;
        color: #e2e8f0;
        margin-top: 0.2rem;
        line-height: 1.5;
        opacity: 0.92;
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
    /* 主对话区：扁平画布，接近 ChatGPT 主区 */
    .block-container > div [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        color: #ececec !important;
        border-radius: 0 !important;
        border: none !important;
        padding: 0.5rem 0.25rem 1rem !important;
        box-shadow: none !important;
    }
    .block-container > div [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0 !important;
        background: transparent !important;
        color: #e2e8f0 !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.2rem 0 !important;
    }
    [data-testid="stChatMessage"] {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0.35rem 0 !important;
        margin-bottom: 0 !important;
        position: relative !important;
    }
    /* 一问一答收紧为一组；新一轮用户提问前加分隔，读起来有「接在上文后面」的层次 */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
        + [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        margin-top: 0.1rem !important;
        padding-top: 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
        + [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        margin-top: 1rem !important;
        padding-top: 0.55rem !important;
        border-top: 1px solid rgba(129, 140, 248, 0.22) !important;
    }
    .block-container [data-testid="stVerticalBlock"]:has([data-testid="stChatMessage"]) {
        border-left: none !important;
        padding-left: 0 !important;
        margin-left: 0 !important;
    }
    [data-testid="stChatMessage"] > div {
        gap: 0.75rem !important;
        align-items: flex-start !important;
    }
    /* 用户：右侧气泡（ChatGPT 式） */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {
        flex-direction: row-reverse !important;
        align-items: flex-end !important;
        justify-content: flex-start !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
        background: #2f2f2f !important;
        border-radius: 1.15rem !important;
        padding: 0.65rem 1rem !important;
        max-width: min(88%, 34rem) !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        box-sizing: border-box !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] li {
        color: #ececec !important;
    }
    /* 助手：弱化头像、正文偏文档流 */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="chatAvatarIcon-assistant"] {
        opacity: 0.85 !important;
        transform: scale(0.92);
    }
    [data-testid="stChatMessage"] > div > div:has([data-testid="stMarkdownContainer"]) {
        flex: 1 1 auto !important;
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
    /* 助手消息正文：无气泡、偏 ChatGPT 正文字色 */
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.1rem 0 0.2rem 0 !important;
        color: #ececec !important;
        font-size: 0.97rem !important;
        line-height: 1.65 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
        color: #ececec !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
        color: #a5b4fc !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
        color: #fde68a !important;
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        padding: 0.08rem 0.35rem !important;
        border-radius: 4px !important;
        font-size: 0.88em !important;
    }
    /* 用户提问：略冷灰，与助手区分，仍非气泡 */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] li {
        color: #cbd5e1 !important;
    }
    [data-testid="stChatMessage"] label,
    [data-testid="stChatMessage"] [data-testid="stCaption"],
    [data-testid="stChatMessage"] .stMarkdown label {
        color: #94a3b8 !important;
    }
    [data-testid="stChatMessage"] [data-baseweb="select"] > div {
        color: #e2e8f0 !important;
    }
    [data-testid="stChatMessage"] pre,
    [data-testid="stChatMessage"] .stCodeBlock {
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        background: rgba(15, 23, 42, 0.55) !important;
        color: #e2e8f0 !important;
    }
    .stCodeBlock { border-radius: 10px !important; border: 1px solid rgba(148, 163, 184, 0.2) !important; }
    /* 流式输出、相关段落展开：与正文同色（不在 chat_message 内时） */
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] li {
        color: #e2e8f0 !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 10px !important;
        background: rgba(15, 23, 42, 0.4) !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpander"] summary,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stExpander"] summary span {
        color: #cbd5e1 !important;
    }
    /* 底部输入：圆角矩形条，接近 ChatGPT dock */
    [data-testid="stChatInput"] {
        background: #2f2f2f !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 1.25rem !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.06), 0 4px 16px rgba(0, 0, 0, 0.45) !important;
    }
    [data-testid="stChatInput"] textarea {
        font-size: 0.95rem !important;
        color: #e2e8f0 !important;
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    [data-testid="stChatInput"] button {
        background: #ececec !important;
        color: #212121 !important;
        border: none !important;
        border-radius: 999px !important;
    }
    /* 消息下方操作条：小图标按钮感 */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stHorizontalBlock"] {
        gap: 0.25rem !important;
        align-items: center !important;
        margin-top: 0.15rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stHorizontalBlock"] button {
        min-height: 2rem !important;
        padding: 0.15rem 0.45rem !important;
        font-size: 0.85rem !important;
        border-radius: 10px !important;
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #d1d5db !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stHorizontalBlock"] button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #f9fafb !important;
    }
    /* 消息内反馈：深色底上的轻量按钮 */
    [data-testid="stChatMessage"] .stButton > button {
        background: rgba(15, 23, 42, 0.45) !important;
        color: #c7d2fe !important;
        border: 1px solid rgba(129, 140, 248, 0.45) !important;
        box-shadow: none !important;
        font-weight: 500 !important;
        min-height: 2.25rem !important;
        padding: 0.2rem 0.75rem !important;
        font-size: 0.8125rem !important;
        border-radius: 8px !important;
    }
    [data-testid="stChatMessage"] .stButton > button:hover {
        background: rgba(79, 70, 229, 0.35) !important;
        border-color: #a5b4fc !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        border-radius: 10px !important;
        background: rgba(15, 23, 42, 0.35) !important;
        margin-top: 0.35rem !important;
    }
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary,
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary span {
        color: #cbd5e1 !important;
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
    /* 主区示例问题：与深色正文区一致（侧栏按钮不受影响：侧栏无 block-container） */
    .block-container [data-testid="stBaseButton-secondary"] {
        background: rgba(15, 23, 42, 0.4) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(148, 163, 184, 0.28) !important;
        font-weight: 500 !important;
        font-size: 0.8125rem !important;
        border-radius: 10px !important;
        box-shadow: none !important;
        min-height: 2.35rem !important;
    }
    .block-container [data-testid="stBaseButton-secondary"]:hover {
        background: rgba(79, 70, 229, 0.28) !important;
        border-color: #818cf8 !important;
        color: #f8fafc !important;
    }
    .meta-box {
        border: 1px solid rgba(148, 163, 184, 0.25);
        background: rgba(15, 23, 42, 0.45);
        border-radius: 12px;
        padding: 0.5rem 0.65rem;
        margin-top: 0.45rem;
        color: #cbd5e1;
    }
    .chip {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        color: #c7d2fe;
        background: rgba(55, 48, 163, 0.45);
        border: 1px solid rgba(129, 140, 248, 0.45);
        border-radius: 999px;
        padding: 0.12rem 0.45rem;
        margin: 0.08rem 0.2rem 0.08rem 0;
    }
    /* 生成中：与正文同一视觉——无浅底盒子，仅浅色字 */
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] {
        background: transparent !important;
        padding: 0.35rem 0 !important;
        border-radius: 0 !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0.25rem 0 0.6rem 0 !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] p,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] span,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stSpinner"] label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] .stSpinner {
        background: transparent !important;
        padding: 0.35rem 0 !important;
        border-radius: 0 !important;
        border: none !important;
        margin: 0.25rem 0 0.6rem 0 !important;
    }
    .block-container [data-testid="stVerticalBlockBorderWrapper"] .stSpinner p,
    .block-container [data-testid="stVerticalBlockBorderWrapper"] .stSpinner span {
        color: #cbd5e1 !important;
    }
    /* 聊天气泡内的带边框容器（来源等）：与深色主题统一 */
    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(148, 163, 184, 0.28) !important;
        background: rgba(15, 23, 42, 0.35) !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"] li {
        color: #cbd5e1 !important;
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


def _render_source_refs_list(source_refs: List[Dict[str, Any]]) -> None:
    for r in source_refs:
        label = r.get("label", "来源")
        hint = r.get("hint", "")
        url = r.get("url", "")
        if url:
            st.markdown(f"- [{label}]({url})  \n  {hint}")
        else:
            st.markdown(f"- {label}  \n  {hint}")


def _render_copy_button(text: str, *, btn_key: str) -> None:
    """用 pyperclip 写入剪贴板，避免 components.html 在部分环境下把脚本当正文显示。"""
    if st.button("复制", key=btn_key, help="复制本条回答到剪贴板"):
        try:
            import pyperclip

            pyperclip.copy(text or "")
        except ImportError:
            st.warning("请安装依赖：pip install pyperclip")
        except Exception:
            st.warning("无法写入剪贴板，请手动选中文字复制。")
        else:
            try:
                st.toast("已复制", icon="✅")
            except Exception:
                st.caption("已复制")


def _regenerate_assistant_message(msg_id: str, messages: List[Dict[str, Any]]) -> bool:
    """删除指定助手消息后，用其对应问题重新发起流式生成。"""
    idx: int | None = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and str(m.get("id")) == str(msg_id):
            idx = i
            break
    if idx is None:
        return False
    q = messages[idx].get("question")
    if not (isinstance(q, str) and q.strip()):
        for j in range(idx - 1, -1, -1):
            if messages[j].get("role") == "user":
                uc = messages[j].get("content")
                if isinstance(uc, str) and uc.strip():
                    q = uc
                break
    if not (isinstance(q, str) and q.strip()):
        return False
    del messages[idx]
    st.session_state["feedback_state"].pop(msg_id, None)
    st.session_state["memory"] = _memory_from_messages(messages)
    st.session_state["_run_stream_for"] = q.strip()
    return True


def _edit_last_user_and_resend(user_idx: int, new_text: str) -> None:
    """截断该用户消息之后的所有轮次，更新用户文本并重新发起流式生成。"""
    msgs = st.session_state["message"]
    if user_idx < 0 or user_idx >= len(msgs):
        return
    if msgs[user_idx].get("role") != "user":
        return
    nt = new_text.strip()
    if not nt:
        return
    for _m in msgs[user_idx + 1 :]:
        _oid = _m.get("id")
        if _oid:
            st.session_state["feedback_state"].pop(_oid, None)
    del msgs[user_idx + 1 :]
    msgs[user_idx]["content"] = nt
    st.session_state["memory"] = _memory_from_messages(msgs)
    _clear_stream_ui_state()
    st.session_state["_run_stream_for"] = nt
    _touch_active_conversation()


def _conversation_to_markdown(title: str, messages: List[Dict[str, Any]]) -> str:
    lines = [
        f"# {title}",
        "",
        f"_导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
    ]
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            label = "用户"
        elif role == "assistant":
            label = "助手"
        else:
            label = str(role or "消息")
        lines.append(f"## {label}")
        lines.append("")
        lines.append(content)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _export_md_filename(title: str) -> str:
    base = re.sub(r'[\\/:*?"<>|\r\n]+', "_", (title or "").strip())
    base = (base[:48] if base else "对话").strip("._") or "对话"
    return f"课程助手_{base}_{time.strftime('%Y%m%d_%H%M%S')}.md"


_BACKUP_VERSION = 1


def _backup_payload_from_session() -> Dict[str, Any]:
    convs = st.session_state.get("conversations")
    if not isinstance(convs, dict) or not convs:
        raise ValueError("无对话数据")
    return {
        "version": _BACKUP_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "active_conv_id": st.session_state.get("active_conv_id"),
        "conversations": json.loads(json.dumps(convs, default=str)),
    }


def _apply_backup_payload(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("备份格式无效")
    if int(data.get("version", -1)) != _BACKUP_VERSION:
        raise ValueError(f"不支持的备份版本：{data.get('version')!r}")
    convs = data.get("conversations")
    if not isinstance(convs, dict) or not convs:
        raise ValueError("备份中无对话列表")
    aid = data.get("active_conv_id")
    if not isinstance(aid, str) or aid not in convs:
        aid = next(iter(convs))
    for _cid, _c in convs.items():
        if not isinstance(_c, dict):
            raise ValueError("对话条目格式错误")
        _msgs = _c.get("messages")
        if not isinstance(_msgs, list):
            raise ValueError("备份中对话缺少 messages")
    st.session_state["conversations"] = convs
    st.session_state["active_conv_id"] = aid
    st.session_state["_conv_switched"] = True
    _clear_stream_ui_state()
    st.session_state["_run_stream_for"] = None
    st.session_state["feedback_state"] = {}


def _render_sidebar_chats() -> None:
    with st.sidebar:
        st.markdown("##### 历史会话")
        if st.button("➕ 新建会话", use_container_width=True, type="primary"):
            new_cid = str(uuid.uuid4())
            new_sid = str(uuid.uuid4())
            welcome = _welcome_messages()
            convs = st.session_state["conversations"]
            _ensure_session_numbers(convs)
            sn = _next_session_no(convs)
            st.session_state["conversations"][new_cid] = {
                "session_no": sn,
                "title": f"会话 {sn}",
                "messages": welcome,
                "session_id": new_sid,
                "updated_at": time.time(),
            }
            st.session_state["active_conv_id"] = new_cid
            st.session_state["_conv_switched"] = True
            _clear_stream_ui_state()
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
                    _clear_stream_ui_state()
                    st.session_state["_run_stream_for"] = None
                    st.rerun()
        _aid = st.session_state.get("active_conv_id")
        _convs = st.session_state.get("conversations")
        if _aid and isinstance(_convs, dict) and _aid in _convs:
            _data = _convs[_aid]
            _title = str(_data.get("title") or "对话")
            _msgs = _data.get("messages")
            if isinstance(_msgs, list) and _msgs:
                _md = _conversation_to_markdown(_title, _msgs)
                st.download_button(
                    "导出当前对话 (.md)",
                    data=_md.encode("utf-8"),
                    file_name=_export_md_filename(_title),
                    mime="text/markdown",
                    use_container_width=True,
                    type="secondary",
                    key="sidebar_export_md",
                )
        with st.expander("备份与恢复", expanded=False):
            st.caption("JSON 含全部会话，可跨设备迁移；恢复会覆盖当前页对话。")
            try:
                _bk = _backup_payload_from_session()
                st.download_button(
                    "导出全部对话备份 (.json)",
                    data=json.dumps(_bk, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"课程助手_全部对话_{time.strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    type="secondary",
                    key="sidebar_export_json",
                )
            except ValueError:
                pass
            _rf = st.file_uploader(
                "从备份恢复 (.json)",
                type=["json"],
                help="选择此前导出的 JSON，点下方按钮确认后将覆盖本页全部对话。",
                key="sidebar_restore_json",
            )
            if _rf is not None:
                st.caption(f"已选择：{_rf.name}")
                if st.button(
                    "确认用此文件恢复",
                    key="sidebar_confirm_restore",
                    type="primary",
                ):
                    try:
                        _apply_backup_payload(
                            json.loads(_rf.getvalue().decode("utf-8"))
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"恢复失败：{e}")
        st.caption(
            "对话仅存于本会话：刷新或关闭页面后通常会清空。"
            " 需要留存时请导出 .md 单条或 JSON 全量备份。"
        )


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
                    "session_no": 1,
                    "title": "会话 1",
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
                    "session_no": 1,
                    "title": "会话 1",
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

    _ensure_session_numbers(st.session_state["conversations"])

    _render_sidebar_chats()

    _conv = st.session_state["conversations"][st.session_state["active_conv_id"]]
    st.session_state["message"] = _conv["messages"]
    st.session_state["session_id"] = _conv["session_id"]

    if st.session_state.pop("_conv_switched", False):
        _clear_stream_ui_state()
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
    <div class="neo-tagline">课程知识库 · 流式回答 · 来源可追溯 · 问答可写入检索库</div>
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

    with st.container():
            # 最新一条助手消息 id（用于「重新生成」仅作用于最后一条，与 ChatGPT 一致）
            last_assistant_id: str | None = None
            for m in reversed(st.session_state["message"]):
                if m.get("role") == "assistant" and m.get("id"):
                    last_assistant_id = str(m["id"])
                    break

            last_user_idx: int | None = None
            for _ui, _um in enumerate(st.session_state["message"]):
                if _um.get("role") == "user":
                    last_user_idx = _ui

            _stream_busy = bool(
                st.session_state.get("_run_stream_for")
            ) or ("_stream_event_iter" in st.session_state)

            for idx, message in enumerate(st.session_state["message"]):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "user":
                        if (
                            last_user_idx is not None
                            and idx == last_user_idx
                            and not _stream_busy
                        ):
                            _eid = st.session_state.get(
                                "active_conv_id", "default"
                            )
                            _ta_key = f"edit_last_user_{_eid}"
                            with st.popover("改提问", help="改字后重新生成本条回答"):
                                _edited = st.text_area(
                                    " ",
                                    value=message.get("content") or "",
                                    height=72,
                                    key=_ta_key,
                                    label_visibility="collapsed",
                                    placeholder="修改问题…",
                                )
                                if st.button(
                                    "重新生成",
                                    key=f"{_ta_key}_submit",
                                    type="primary",
                                    use_container_width=True,
                                ):
                                    _nt = (_edited or "").strip()
                                    if not _nt:
                                        st.warning("内容不能为空")
                                    else:
                                        _edit_last_user_and_resend(idx, _nt)
                                        st.rerun()
                    if message["role"] == "assistant" and message.get("content"):
                        _meta = message.get("meta")
                        _refs: List[Dict[str, Any]] = []
                        if isinstance(_meta, dict):
                            _r = _meta.get("source_refs")
                            if isinstance(_r, list):
                                _refs = [x for x in _r if isinstance(x, dict)]
                        if _refs:
                            with st.expander(
                                f"来源（{len(_refs)}）",
                                expanded=False,
                            ):
                                _render_source_refs_list(_refs)
                        msg_id = message.get("id")
                        if msg_id:
                            _msg_id_s = str(msg_id)
                            fb = st.session_state["feedback_state"].get(
                                msg_id, {}
                            )
                            submitted = fb.get("submitted", False)
                            if submitted:
                                st.caption("反馈已提交，感谢你的反馈。")
                            else:
                                ac1, ac2, ac3, ac4 = st.columns(
                                    [1.4, 0.7, 0.7, 0.7]
                                )
                                with ac1:
                                    _render_copy_button(
                                        message.get("content") or "",
                                        btn_key=f"copy_{_msg_id_s}",
                                    )
                                with ac2:
                                    if st.button(
                                        "👍",
                                        key=f"fb_up_{_msg_id_s}",
                                        help="有帮助",
                                    ):
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
                                with ac3:
                                    with st.popover("👎"):
                                        st.caption("请选择原因后提交")
                                        down_reason = st.selectbox(
                                            "原因",
                                            options=[
                                                "答非所问",
                                                "不准确",
                                                "信息不足",
                                                "太慢",
                                                "其他",
                                            ],
                                            index=0,
                                            key=f"fb_reason_{_msg_id_s}",
                                            label_visibility="collapsed",
                                        )
                                        if st.button(
                                            "提交差评",
                                            key=f"fb_down_{_msg_id_s}",
                                        ):
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
                                                st.warning(
                                                    f"反馈提交失败: {err}"
                                                )
                                with ac4:
                                    if (
                                        last_assistant_id
                                        and _msg_id_s == last_assistant_id
                                        and not st.session_state.get(
                                            "_run_stream_for"
                                        )
                                    ):
                                        if st.button(
                                            "↻",
                                            key=f"regen_{_msg_id_s}",
                                            help="重新生成",
                                        ):
                                            if _regenerate_assistant_message(
                                                _msg_id_s,
                                                st.session_state["message"],
                                            ):
                                                _touch_active_conversation()
                                                st.rerun()

            pending = st.session_state.get("_run_stream_for")
            if pending:
                stream_prompt = pending
                history_payload = _api_chat_history(
                    st.session_state["message"]
                )
                # 分块消费 SSE + 多次 rerun，才能在本页渲染「停止生成」并关闭连接
                if "_stream_event_iter" not in st.session_state:
                    try:
                        resp = _chat_stream_post(
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
                        )
                        st.session_state["_stream_resp"] = resp
                        st.session_state["_stream_event_iter"] = iter(
                            _parse_sse_lines(resp)
                        )
                        st.session_state["_stream_acc"] = {
                            "context_text": "",
                            "current_answer": "",
                            "route_meta": None,
                            "spinner_cleared": False,
                        }
                    except requests.RequestException as e:
                        _clear_stream_ui_state()
                        err = f"后端请求失败：{e}"
                        st.session_state["message"].append(
                            {
                                "id": str(uuid.uuid4()),
                                "role": "assistant",
                                "content": err,
                                "question": stream_prompt,
                                "meta": None,
                            }
                        )
                        st.session_state["memory"].chat_memory.add_ai_message(
                            err
                        )
                        _touch_active_conversation()
                        st.session_state["_run_stream_for"] = None
                        st.rerun()

                acc = st.session_state.get("_stream_acc")
                if acc is None:
                    st.session_state["_run_stream_for"] = None
                    st.rerun()

                thinking_slot = st.empty()
                context_placeholder = st.empty()
                answer_area = st.empty()

                if not acc.get("spinner_cleared"):
                    with thinking_slot:
                        with st.spinner("AI 正在思考中"):
                            st.caption(" ")
                else:
                    thinking_slot.empty()

                if st.button(
                    "停止",
                    key="stop_stream_gen",
                    type="secondary",
                    help="停止生成当前回答",
                    use_container_width=True,
                ):
                    ca = (acc.get("current_answer") or "").strip()
                    ctx = (acc.get("context_text") or "").strip()
                    rm = acc.get("route_meta")
                    _clear_stream_ui_state()
                    if ca:
                        final = ca + "\n\n（已停止生成）"
                    elif ctx:
                        final = ctx + "\n\n（已停止生成）"
                    else:
                        final = "（已停止生成）"
                    st.session_state["message"].append(
                        {
                            "id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": final,
                            "question": stream_prompt,
                            "meta": rm,
                        }
                    )
                    st.session_state["memory"].chat_memory.add_ai_message(
                        final
                    )
                    if rm:
                        st.session_state["last_route_meta"] = rm
                    _touch_active_conversation()
                    st.session_state["_run_stream_for"] = None
                    st.rerun()

                it = st.session_state["_stream_event_iter"]
                stream_done = False
                for _ in range(_STREAM_BATCH_SIZE):
                    try:
                        event = next(it)
                    except StopIteration:
                        stream_done = True
                        break
                    event_type = event.get("type")
                    event_content = event.get("content") or ""
                    if event_type == "context":
                        acc["context_text"] = event_content
                        if (
                            event_content.strip()
                            and event_content.strip()
                            != "没有检索到所需内容"
                        ):
                            acc["spinner_cleared"] = True
                            with context_placeholder.container():
                                with st.expander(
                                    "相关摘录",
                                    expanded=False,
                                ):
                                    st.markdown(event_content)
                        else:
                            context_placeholder.empty()
                    elif event_type == "answer":
                        acc["spinner_cleared"] = True
                        acc["current_answer"] = event_content
                        answer_area.markdown(event_content)
                    elif event_type == "meta":
                        if isinstance(event_content, dict):
                            acc["route_meta"] = event_content
                    elif event_type == "done":
                        if (
                            not (acc.get("current_answer") or "").strip()
                            and event_content
                        ):
                            acc["current_answer"] = event_content
                            answer_area.markdown(event_content)
                        acc["spinner_cleared"] = True
                        stream_done = True
                        break
                    elif event_type == "error":
                        acc["spinner_cleared"] = True
                        acc["current_answer"] = f"错误：{event_content}"
                        answer_area.markdown(acc["current_answer"])
                        stream_done = True
                        break

                if not stream_done:
                    st.rerun()

                current_answer = acc.get("current_answer") or ""
                context_text = acc.get("context_text") or ""
                route_meta = acc.get("route_meta")

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
                    if isinstance(source_refs, list):
                        source_refs = [
                            x for x in source_refs if isinstance(x, dict)
                        ]
                    else:
                        source_refs = []
                    if source_refs:
                        with st.expander(
                            f"来源（{len(source_refs)}）",
                            expanded=False,
                        ):
                            _render_source_refs_list(source_refs)
                    if route_meta.get("degraded"):
                        suggestions = route_meta.get("suggestions") or []
                        if suggestions:
                            with st.expander("参考问法", expanded=False):
                                for s in suggestions[:3]:
                                    st.markdown(f"- {s}")

                if not current_answer.strip() and use_llm is False:
                    current_answer = context_text
                if not current_answer.strip():
                    if not use_llm:
                        current_answer = context_text or "未检索到相关段落。"
                    else:
                        current_answer = (
                            "未获取到模型输出（请检查后端/网络/密钥）。"
                        )

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
                _clear_stream_ui_state()
                st.session_state["_run_stream_for"] = None
                st.rerun()

    prompt = st.chat_input("有问题，尽管问…")
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
