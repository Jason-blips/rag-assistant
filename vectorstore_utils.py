from pathlib import Path
import pickle
from typing import Iterable, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# 基础配置集中在这里，便于统一修改
DEFAULT_KNOWLEDGE_BASE_DIR = Path("knowledge_base")  # 课件统一存放目录
DEFAULT_PDF_PATH = DEFAULT_KNOWLEDGE_BASE_DIR / "data_week6.pdf"
DEFAULT_PERSIST_DIR = Path("./vector_db")
# 为了兼容你之前已经构建好的向量库，这里保持 Chroma 的默认 collection 名称：langchain
DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
BM25_CORPUS_FILENAME = "bm25_corpus.pkl"
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.6
RRF_K = 60  # Reciprocal Rank Fusion 常数


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    创建并返回 HuggingFaceEmbeddings。
    通过集中入口保证构建阶段与查询阶段使用完全一致的模型配置。
    """

    name = model_name or DEFAULT_EMBEDDING_MODEL_NAME
    kwargs: dict = {}
    if "bge" in name.lower():
        kwargs["encode_kwargs"] = {"normalize_embeddings": True}
        kwargs["query_instruction"] = "为这个句子生成表示以用于检索相关文章："
    return HuggingFaceEmbeddings(model_name=name, **kwargs)


def _bm25_corpus_path(persist_directory: Path | str) -> Path:
    return Path(persist_directory) / BM25_CORPUS_FILENAME


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


def hybrid_retrieve(
    db: Chroma,
    query: str,
    k: int = 3,
    candidate_k: int = 15,
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
) -> tuple[list[Document], list[float]]:
    """
    BM25 关键词 + 向量语义双路检索，用 Reciprocal Rank Fusion (RRF) 融合排序。

    返回 (docs, rrf_scores)，长度为 min(k, 去重后候选数)。
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

    for rank, (doc, _score) in enumerate(vector_pairs):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + vector_weight / (RRF_K + rank + 1)
        doc_map.setdefault(key, doc)

    for rank, (doc, _score) in enumerate(bm25_pairs):
        key = doc.page_content
        rrf_scores[key] = rrf_scores.get(key, 0.0) + bm25_weight / (RRF_K + rank + 1)
        doc_map.setdefault(key, doc)

    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:k]
    result_docs = [doc_map[key] for key in sorted_keys]
    result_scores = [rrf_scores[key] for key in sorted_keys]
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

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    docs = text_splitter.split_documents(documents)

    if extra_metadata:
        for d in docs:
            d.metadata.update(extra_metadata)

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
) -> Chroma:
    """
    从指定 PDF 构建并（可能是增量）写入 Chroma 向量库。

    如果向量库已经存在，会在原有基础上追加这个 PDF 的内容；
    这更适合“多个课件组成一个大知识库”的场景。
    """
    pdf_path = Path(pdf_path)
    extra_meta = {
        "source": str(pdf_path),
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
    return db


def build_vectorstore_from_pdf_dir(
    pdf_root_dir: Path | str,
    persist_directory: Path | str = DEFAULT_PERSIST_DIR,
    model_name: Optional[str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
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
    all_texts: list[Document] = []
    count = 0
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            continue

        try:
            rel = pdf_path.relative_to(pdf_root_dir)
            parts = rel.parts
            course_name = parts[0] if len(parts) > 1 else pdf_path.parent.name
        except ValueError:
            course_name = pdf_path.parent.name

        extra_meta = {
            "source": str(pdf_path),
            "course": course_name,
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

