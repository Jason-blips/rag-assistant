from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pickle
from typing import Iterable, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# 基础配置集中在这里，便于统一修改
# 注意：不要使用当前工作目录（cwd）相关的相对路径，否则从不同目录启动脚本会导致找不到文件。
_PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_KNOWLEDGE_BASE_DIR = _PROJECT_ROOT / "knowledge_base"  # 课件统一存放目录
DEFAULT_PDF_PATH = DEFAULT_KNOWLEDGE_BASE_DIR / "data_week6.pdf"
DEFAULT_PERSIST_DIR = _PROJECT_ROOT / "vector_db"
# 为了兼容你之前已经构建好的向量库，这里保持 Chroma 的默认 collection 名称：langchain
DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_CONVERSATION_COLLECTION_NAME = "conversation_memory"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
BM25_CORPUS_FILENAME = "bm25_corpus.pkl"
KNOWLEDGE_MANIFEST_FILENAME = "knowledge_manifest.json"
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.6
RRF_K = 60  # Reciprocal Rank Fusion 常数
DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


def infer_topic(text: str, *, block_type: str | None = None) -> str:
    """
    轻量启发式：根据文本内容推断属于哪个知识模块。

    该函数用于：
    - 构建向量库时给 chunk 写入 topic 元数据
    - 检索时根据用户问题推断 topic_preference 并偏置候选
    """
    s = (text or "").strip()
    if not s:
        return "general"

    lower = s.lower()

    # block_type 直接给一些先验（例如 code/table）
    if block_type == "code":
        return "code"
    if block_type == "table":
        return "table"

    def _has_any(keywords: list[str]) -> int:
        return sum(1 for k in keywords if k in lower)

    # 依据课程常见内容做关键词计分
    uml_score = _has_any(
        [
            "uml",
            "class diagram",
            "类图",
            "association",
            "inheritance",
            "generalization",
            "aggregation",
            "composition",
            "interface",
            "multiplicity",
            "object oriented",
        ]
    )
    sql_score = _has_any(
        [
            "select",
            "from",
            "where",
            "join",
            "group by",
            "order by",
            "create table",
            "insert",
            "update",
            "delete",
            "primary key",
            "foreign key",
            "sql",
            "数据库",
            "表",
        ]
    )
    er_score = _has_any(
        [
            "er",
            "entity",
            "relationship",
            "实体",
            "关系",
            "实体-关系",
        ]
    )
    algo_score = _has_any(
        [
            "algorithm",
            "algorithms",
            "time complexity",
            "space complexity",
            "big o",
            "big-o",
            "o(",
            "o ",
            "复杂度",
            "渐进",
            "big",
        ]
    )

    # 兜底：取最高分
    scores = {
        "uml": uml_score,
        "sql": sql_score,
        "er": er_score,
        "algorithm": algo_score,
    }
    topic = max(scores, key=scores.__getitem__)
    if scores[topic] <= 0:
        return "general"
    return topic


def compute_course_match_score(question: str) -> tuple[float, str]:
    """
    计算“问题与课程课件的匹配度”，用于决定是否触发向量检索上下文。

    返回：
    - match_score: 0~1
    - inferred_topic: infer_topic(question) 的结果
    """
    s = (question or "").strip()
    if not s:
        return 0.0, "general"

    lower = s.lower()
    topic = infer_topic(s)

    # 学术/计算机课程常见信号词：用于在 topic="general" 时仍能给出匹配度
    academic_keywords = [
        "algorithm",
        "algorithms",
        "complexity",
        "time complexity",
        "space complexity",
        "big o",
        "big-o",
        "o(",
        "o ",
        "复杂度",
        "算法",
        "渐进",
        "证明",
        "定理",
        "引理",
        "推导",
        "评分",
        "作业",
        "标准",
        "rubric",
        "assignment",
        "exam",
        "考试",
        "题目",
    ]

    topic_keywords: dict[str, list[str]] = {
        "uml": [
            "uml",
            "类图",
            "association",
            "inheritance",
            "generalization",
            "aggregation",
            "composition",
            "interface",
            "multiplicity",
            "object oriented",
        ],
        "sql": [
            "sql",
            "select",
            "from",
            "where",
            "join",
            "group by",
            "order by",
            "create table",
            "insert",
            "update",
            "delete",
            "database",
            "表",
        ],
        "er": [
            "er",
            "entity",
            "relationship",
            "实体",
            "关系",
        ],
        "algorithm": [
            "algorithm",
            "complexity",
            "time complexity",
            "space complexity",
            "big o",
            "big-o",
            "复杂度",
            "算法",
        ],
    }

    def _count_hits(keys: list[str]) -> int:
        hits = 0
        for k in keys:
            if k in lower:
                hits += 1
        return hits

    hit_academic = _count_hits(academic_keywords)

    if topic != "general":
        keys = topic_keywords.get(topic, [])
        hit_topic = _count_hits(keys)
        # topic 检测已命中至少一个关键词：给一个“底线分”，否则 hit=1 时会过于苛刻。
        base = 0.3
        extra = (max(0, hit_topic - 1) / 4) * 0.7
        match_score = min(1.0, base + extra)
        return match_score, topic

    # topic="general"：只靠学术信号词打分
    match_score = min(1.0, hit_academic / 5)
    return match_score, topic


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    创建并返回 HuggingFaceEmbeddings。
    通过集中入口保证构建阶段与查询阶段使用完全一致的模型配置。
    """

    name = model_name or DEFAULT_EMBEDDING_MODEL_NAME
    kwargs: dict = {}
    if "bge" in name.lower():
        kwargs["encode_kwargs"] = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(model_name=name, **kwargs)


def _bm25_corpus_path(persist_directory: Path | str) -> Path:
    return Path(persist_directory) / BM25_CORPUS_FILENAME


def _knowledge_manifest_path(persist_directory: Path | str) -> Path:
    return Path(persist_directory) / KNOWLEDGE_MANIFEST_FILENAME


def _compute_knowledge_version(pdf_paths: list[Path]) -> str:
    digest = hashlib.sha1()
    for p in sorted(pdf_paths, key=lambda x: str(x).lower()):
        stat = p.stat()
        digest.update(str(p.resolve()).encode("utf-8", errors="ignore"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()[:12]


def write_knowledge_manifest(
    *,
    persist_directory: Path | str,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    pdf_paths: list[Path],
    chunk_count: int,
    knowledge_version: str | None = None,
) -> dict:
    built_at = datetime.now(timezone.utc).isoformat()
    version = knowledge_version or _compute_knowledge_version(pdf_paths)
    data = {
        "knowledge_version": version,
        "built_at": built_at,
        "pdf_count": len(pdf_paths),
        "chunk_count": chunk_count,
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "files": [str(p.resolve()) for p in sorted(pdf_paths, key=lambda x: str(x).lower())],
    }
    path = _knowledge_manifest_path(persist_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def load_knowledge_manifest(persist_directory: Path | str = DEFAULT_PERSIST_DIR) -> dict | None:
    path = _knowledge_manifest_path(persist_directory)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_bm25_corpus(
    docs: list[Document],
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    append: bool = False,
) -> None:
    """将切分后的 Document 列表序列化，供 BM25 检索使用。"""
    path = _bm25_corpus_path(persist_directory)
    path.parent.mkdir(parents=True, exist_ok=True)

    if append and path.exists():
        with open(path, "rb") as f:
            existing: list[Document] = pickle.load(f)
        existing.extend(docs)
        docs = existing

    with open(path, "wb") as f:
        pickle.dump(docs, f)


def load_bm25_corpus(persist_directory: Path | str = DEFAULT_PERSIST_DIR) -> list[Document]:
    path = _bm25_corpus_path(persist_directory)
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 语料库不存在: {path.resolve()}，请先运行 build_vector_db.py 构建。"
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：jieba 切中文，空格切英文，全部转小写。"""
    import jieba
    tokens: list[str] = []
    for token in jieba.cut(text):
        t = token.strip().lower()
        if t:
            tokens.append(t)
    return tokens


def rerank(
    query: str,
    docs: list[Document],
    top_k: int = 3,
    model_name: str = DEFAULT_RERANKER_MODEL_NAME,
) -> tuple[list[Document], list[float]]:
    """
    用 Cross-Encoder 对候选文档精排，返回 top_k 结果和对应分数。
    模型首次调用时自动下载并缓存。
    """
    from sentence_transformers import CrossEncoder

    if not docs:
        return [], []

    model = CrossEncoder(model_name)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs).tolist()

    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [d for d, _ in scored], [s for _, s in scored]


def hybrid_retrieve(
    db: Chroma,
    query: str,
    k: int = 3,
    candidate_k: int = 15,
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    use_reranker: bool = True,
    reranker_model: str = DEFAULT_RERANKER_MODEL_NAME,
    topic_preference: str | None = None,
    topic_mismatch_weight: float = 0.3,
) -> tuple[list[Document], list[float]]:
    """
    BM25 关键词 + 向量语义双路检索，RRF 融合，可选 Cross-Encoder 精排。

    流程：双路各取 candidate_k 条 → RRF 融合去重 → (可选) Reranker 精排 → 返回 top-k。
    """
    from rank_bm25 import BM25Okapi

    # --- 向量检索 ---
    vector_pairs = db.similarity_search_with_relevance_scores(query, k=candidate_k)

    # --- BM25 检索 ---
    corpus_docs = load_bm25_corpus(persist_directory)
    tokenized_corpus = [_tokenize(d.page_content) for d in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:candidate_k]
    bm25_pairs = [(corpus_docs[i], bm25_scores[i]) for i in top_indices]

    # --- RRF 融合 ---
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    def _topic_factor(doc: Document) -> float:
        if not topic_preference:
            return 1.0
        md = getattr(doc, "metadata", None) or {}
        # 会话记忆要始终参与检索偏置；否则对话上下文容易被“topic”误伤。
        if md.get("doc_type") == "conversation":
            return 1.0
        doc_topic = md.get("topic")
        return 1.0 if doc_topic == topic_preference else topic_mismatch_weight

    for rank, (doc, _score) in enumerate(vector_pairs):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (
            vector_weight * _topic_factor(doc) / (RRF_K + rank + 1)
        )
        doc_map.setdefault(key, doc)

    for rank, (doc, _score) in enumerate(bm25_pairs):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (
            bm25_weight * _topic_factor(doc) / (RRF_K + rank + 1)
        )
        doc_map.setdefault(key, doc)

    # RRF 粗排：取足够多的候选送给 reranker
    rerank_pool_size = max(k * 3, 10) if use_reranker else k
    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:rerank_pool_size]
    candidates = [doc_map[key] for key in sorted_keys]

    # --- Reranker 精排 ---
    if use_reranker and candidates:
        return rerank(query=query, docs=candidates, top_k=k, model_name=reranker_model)

    result_docs = candidates[:k]
    result_scores = [rrf_scores[d.page_content] for d in result_docs]
    return result_docs, result_scores


def get_text_chunks_from_pdf(
    pdf_path: Path | str = DEFAULT_PDF_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    extra_metadata: Optional[dict] = None,
) -> list[Document]:
    """
    从 PDF 路径加载文档并切分为文本块。
    支持附加自定义 metadata（例如课程名、章节等），方便后续按课程检索与溯源。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_path.resolve()}")

    def _cjk_count(s: str) -> int:
        import re

        return len(re.findall(r"[\u4e00-\u9fff]", s))

    def _is_code_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        code_markers = [
            "```",
            "{",
            "}",
            ";",
            "->",
            "=>",
            "SELECT",
            "FROM",
            "WHERE",
            "CREATE",
            "DROP",
            "JOIN",
            "GROUP BY",
            "ORDER BY",
            "class ",
            "def ",
            "import ",
            "public ",
            "private ",
            "void ",
            "int ",
            "float ",
            "return ",
            "==",
            "!=",
            "<=",
            ">=",
        ]
        s_upper = s.upper()
        if any(m in s for m in code_markers):
            return True
        if any(m in s_upper for m in code_markers if m.isupper()):
            return True
        # 代码块常见缩进
        if line.startswith(("    ", "\t")):
            return True
        return False

    def _is_table_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if "|" in s:
            return True
        import re

        # 连续多空格常见于 PDF 导出的表格列
        if re.search(r"\s{2,}", line) is not None:
            parts = [p for p in re.split(r"\s{2,}", s) if p.strip()]
            return len(parts) >= 3
        return False

    def _is_heading_line(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if _is_code_line(s) or _is_table_line(s):
            return False
        # Markdown 标题（如果 PDF 恰好保留了）
        if s.startswith("#"):
            return True
        # 常见“第X章/第X节”标题
        import re

        if re.match(r"^第[一二三四五六七八九十0-9]+[章节条].*$", s):
            return True
        # 短句且不以句号/问号结尾，常是标题
        if len(s) <= 80 and not s.endswith(("。", "？", "！", ".", "?", "!", ":")):
            if _cjk_count(s) > 0:
                return True
            # 英文标题倾向于 Title Case 或包含少量符号
            if re.match(r"^[A-Za-z][A-Za-z0-9\s\\-]+$", s):
                return True
        return False

    def _split_structured_text(text: str) -> list[tuple[str, str]]:
        """
        将一段文本拆成若干块：普通(normal) / 代码(code) / 表格(table)。
        """
        lines = text.splitlines()
        blocks: list[tuple[str, list[str]]] = []
        buf: list[str] = []
        buf_type: str = "normal"
        miss_count = 0

        def _flush():
            nonlocal buf, buf_type, miss_count
            if buf:
                blocks.append((buf_type, buf))
            buf = []
            buf_type = "normal"
            miss_count = 0

        in_fence = False
        for line in lines:
            # fence 形式代码块（如果 PDF 恰好保留了 ```）
            if "```" in line:
                if not in_fence:
                    _flush()
                    buf_type = "code"
                    buf.append(line)
                    in_fence = True
                else:
                    buf.append(line)
                    in_fence = False
                    _flush()
                continue

            if in_fence:
                buf_type = "code"
                buf.append(line)
                continue

            if buf_type == "normal":
                if _is_code_line(line):
                    buf_type = "code"
                    buf.append(line)
                elif _is_table_line(line):
                    buf_type = "table"
                    buf.append(line)
                else:
                    buf.append(line)
            else:
                # code/table 块允许遇到空行
                if not line.strip():
                    buf.append(line)
                    continue

                if buf_type == "code" and _is_code_line(line):
                    buf.append(line)
                    continue
                if buf_type == "table" and _is_table_line(line):
                    buf.append(line)
                    continue

                miss_count += 1
                if miss_count >= 2:
                    _flush()
                    buf.append(line)
                else:
                    buf.append(line)

        _flush()
        return [(t, "\n".join(ls).strip()) for t, ls in blocks if "\n".join(ls).strip()]

    def _split_by_heading_then_recursive(
        text: str, *, separators: list[str]
    ) -> list[str]:
        """
        对“普通文本块”再做一次疑似标题分段，然后用 RecursiveCharacterTextSplitter 兜底。
        """
        lines = text.splitlines()
        segments: list[str] = []
        cur: list[str] = []
        for line in lines:
            if _is_heading_line(line) and cur:
                seg = "\n".join(cur).strip()
                if seg:
                    segments.append(seg)
                cur = [line]
            else:
                cur.append(line)
        tail = "\n".join(cur).strip()
        if tail:
            segments.append(tail)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
        )

        out: list[str] = []
        for seg in segments:
            if len(seg) <= chunk_size:
                out.append(seg)
            else:
                out.extend(splitter.split_text(seg))
        return out

    def _split_block_text(block_text: str, block_type: str) -> list[str]:
        if not block_text:
            return []
        if block_type in {"code", "table"}:
            # 保持行边界：优先按换行切，尽量不把表格/代码“切散成碎片”
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n", " "],
            )
            return splitter.split_text(block_text)

        # 普通文本：优先按标题/段落边界拆
        return _split_by_heading_then_recursive(
            block_text,
            separators=["\n\n", "\n", " ", ""],
        )

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    docs: list[Document] = []
    for d in documents:
        base_meta = dict(d.metadata or {})
        if extra_metadata:
            base_meta.update(extra_metadata)

        blocks = _split_structured_text(d.page_content)
        for block_type, block_text in blocks:
            for chunk_text in _split_block_text(block_text, block_type):
                if chunk_text.strip():
                    topic = infer_topic(
                        chunk_text,
                        block_type=block_type,
                    )
                    meta = dict(base_meta)
                    meta.update(
                        {
                            "doc_type": "kb",
                            "topic": topic,
                            "block_type": block_type,
                        }
                    )
                    docs.append(
                        Document(page_content=chunk_text, metadata=meta)
                    )

    return docs


def _ensure_vectorstore(
    persist_directory: Path | str,
    model_name: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """
    初始化或加载一个 Chroma 向量库实例。
    - 如果目录不存在会自动创建；
    - 如果已有数据则在原有 collection 上增量追加。
    """
    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    embeddings = get_embeddings(model_name=model_name)
    db = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    return db


def build_vectorstore_from_pdf(
    pdf_path: Path | str = DEFAULT_PDF_PATH,
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    model_name: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    course_name: Optional[str] = None,
    knowledge_version: str | None = None,
) -> Chroma:
    """
    从指定 PDF 构建并（可能是增量）写入 Chroma 向量库。

    如果向量库已经存在，会在原有基础上追加这个 PDF 的内容；
    这更适合“多个课件组成一个大知识库”的场景。
    """
    pdf_path = Path(pdf_path)
    version = knowledge_version or _compute_knowledge_version([pdf_path])
    extra_meta = {
        "source": str(pdf_path),
        "knowledge_version": version,
    }
    if course_name:
        extra_meta["course"] = course_name

    texts = get_text_chunks_from_pdf(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        extra_metadata=extra_meta,
    )

    db = _ensure_vectorstore(
        persist_directory=persist_directory,
        model_name=model_name,
        collection_name=collection_name,
    )
    db.add_documents(texts)
    save_bm25_corpus(texts, persist_directory=persist_directory, append=True)
    write_knowledge_manifest(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=model_name or DEFAULT_EMBEDDING_MODEL_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pdf_paths=[pdf_path],
        chunk_count=len(texts),
        knowledge_version=version,
    )
    return db


def build_vectorstore_from_pdf_dir(
    pdf_root_dir: Path | str,
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    model_name: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    knowledge_version: str | None = None,
) -> Chroma:
    """
    从一个目录下批量导入多个 PDF，构建/增量更新课程知识库。

    约定：
    - pdf_root_dir/课程名/xxx.pdf  -> metadata.course = 课程名
    - 否则，course 默认为 pdf 文件所在目录名
    """
    pdf_root_dir = Path(pdf_root_dir)
    if not pdf_root_dir.exists():
        raise FileNotFoundError(f"未找到 PDF 根目录: {pdf_root_dir.resolve()}")

    db = _ensure_vectorstore(
        persist_directory=persist_directory,
        model_name=model_name,
        collection_name=collection_name,
    )

    pdf_paths: Iterable[Path] = pdf_root_dir.rglob("*.pdf")
    pdf_file_list = [p for p in pdf_paths if p.is_file()]
    version = knowledge_version or _compute_knowledge_version(pdf_file_list)
    all_texts: list[Document] = []
    count = 0
    for pdf_path in pdf_file_list:

        try:
            rel = pdf_path.relative_to(pdf_root_dir)
            parts = rel.parts
            course_name = parts[0] if len(parts) > 1 else pdf_path.parent.name
        except ValueError:
            course_name = pdf_path.parent.name

        extra_meta = {
            "source": str(pdf_path),
            "course": course_name,
            "knowledge_version": version,
        }

        texts = get_text_chunks_from_pdf(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            extra_metadata=extra_meta,
        )
        if not texts:
            continue

        db.add_documents(texts)
        all_texts.extend(texts)
        count += 1

    if count == 0:
        raise RuntimeError(f"在目录 {pdf_root_dir.resolve()} 下未找到任何 PDF 文件。")

    save_bm25_corpus(all_texts, persist_directory=persist_directory, append=False)
    write_knowledge_manifest(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=model_name or DEFAULT_EMBEDDING_MODEL_NAME,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pdf_paths=pdf_file_list,
        chunk_count=len(all_texts),
        knowledge_version=version,
    )
    return db


def load_vectorstore(
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    model_name: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Chroma:
    """
    加载已经构建好的 Chroma 向量库。
    如果目录不存在，直接抛出带有清晰提示的信息。
    """
    persist_directory = Path(persist_directory)
    if not persist_directory.exists():
        raise FileNotFoundError(
            f"向量库目录不存在: {persist_directory.resolve()}，"
            f"请先运行构建脚本生成向量数据库。"
        )

    embeddings = get_embeddings(model_name=model_name)
    return Chroma(
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

