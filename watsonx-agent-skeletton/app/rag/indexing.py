"""Document loading, splitting, embedding, and storage pipeline.

Documents are fetched from IBM Cloud Object Storage, then split and
stored in the Milvus vector database.
"""

from __future__ import annotations

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.vectorstore import get_vectorstore
from app.storage import download_objects


def index_documents(
    cos_prefix: str,
    collection_name: str | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> int:
    """Download documents from COS, split, embed, and store in Milvus.

    Parameters
    ----------
    cos_prefix:
        Object key prefix in the configured COS bucket.
    collection_name:
        Milvus collection to index into (defaults to config value).
    chunk_size:
        Maximum characters per chunk.
    chunk_overlap:
        Overlap between consecutive chunks.

    Returns
    -------
    int
        The number of document chunks indexed.
    """
    tmp_dir = download_objects(prefix=cos_prefix)

    loader = DirectoryLoader(
        str(tmp_dir), glob="**/*.*", loader_cls=TextLoader, show_progress=True
    )
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)

    vectorstore = get_vectorstore(collection_name)
    vectorstore.add_documents(chunks)

    return len(chunks)
