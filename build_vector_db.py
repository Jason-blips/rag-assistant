from pathlib import Path
import argparse

from vectorstore_utils import (
    build_vectorstore_from_pdf,
    build_vectorstore_from_pdf_dir,
    DEFAULT_PDF_PATH,
    DEFAULT_KNOWLEDGE_BASE_DIR,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_EMBEDDING_MODEL_NAME,
)


def _ensure_utf8_stdout() -> None:
    try:
        import sys

        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> None:
    """
    从指定 PDF 构建 Chroma 向量数据库并持久化到磁盘。
    把原来脚本顶层的执行逻辑收敛到一个函数里，便于复用和测试。
    """
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="从 PDF 构建并持久化 Chroma 向量数据库")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="要构建向量库的单个 PDF 路径（指定后只处理该文件）",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=str(DEFAULT_KNOWLEDGE_BASE_DIR),
        help=f"批量导入的 PDF 根目录（默认: {DEFAULT_KNOWLEDGE_BASE_DIR}，递归查找 *.pdf）",
    )
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"切分块大小（默认: {DEFAULT_CHUNK_SIZE}）",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"切分重叠大小（默认: {DEFAULT_CHUNK_OVERLAP}）",
    )

    args = parser.parse_args()

    persist_dir = Path(args.persist_dir)

    if args.pdf:
        # 指定了单个 PDF —— 只处理这一个文件
        pdf_path = Path(args.pdf)
        build_vectorstore_from_pdf(
            pdf_path=pdf_path,
            persist_directory=persist_dir,
            model_name=args.embedding_model,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            course_name=pdf_path.parent.name,
        )
        print(f"✅ 向量数据库构建/更新完成！")
        print(f"   PDF: {pdf_path.resolve()}")
    else:
        # 默认：从知识库文件夹批量导入
        pdf_root = Path(args.pdf_dir)
        build_vectorstore_from_pdf_dir(
            pdf_root_dir=pdf_root,
            persist_directory=persist_dir,
            model_name=args.embedding_model,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(f"✅ 向量数据库批量构建/更新完成！")
        print(f"   知识库目录: {pdf_root.resolve()}")
    print(f"   向量库目录: {persist_dir.resolve()}")
    print(f"   Collection: {args.collection}")
    print(f"   Embedding: {args.embedding_model}")


if __name__ == "__main__":
    main()
