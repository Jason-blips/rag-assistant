from pathlib import Path
from typing import Iterable, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


# 基础配置集中在这里，便于统一修改
DEFAULT_PDF_PATH = Path("data_week6.pdf")
DEFAULT_PERSIST_DIR = Path("./vector_db")
# 为了兼容你之前已经构建好的向量库，这里保持 Chroma 的默认 collection 名称：langchain
DEFAULT_COLLECTION_NAME = "langchain"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """
    创建并返回 HuggingFaceEmbeddings。
    通过集中入口保证构建阶段与查询阶段使用完全一致的模型配置。
    """

    return HuggingFaceEmbeddings(
        model_name=model_name or DEFAULT_EMBEDDING_MODEL_NAME
    )


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
    count = 0
    for pdf_path in pdf_paths:
        if not pdf_path.is_file():
            continue

        # 课程名：优先取相对根目录的第一级子目录
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
        count += 1

    if count == 0:
        raise RuntimeError(f"在目录 {pdf_root_dir.resolve()} 下未找到任何 PDF 文件。")

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

