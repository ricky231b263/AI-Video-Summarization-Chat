import os
import hashlib
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def _collection_name_for(transcript: str) -> str:
    h = hashlib.sha1(transcript.encode("utf-8")).hexdigest()[:12]
    return f"transcript_{h}"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(transcript: str) -> Chroma:
    print("Building vector store ...")
    collection_name = _collection_name_for(transcript)

    try:
        existing = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        existing.delete_collection()
    except Exception:
        pass

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)

    docs = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )
    return vector_store


def load_vector_store(transcript: str = None) -> Chroma:
    if transcript is None:
        raise ValueError(
            "load_vector_store needs a transcript (or a known collection name) "
            "to know which collection to load."
        )
    collection_name = _collection_name_for(transcript)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def delete_vector_store(transcript: str):
    """Deletes the Chroma collection created for this transcript."""
    collection_name = _collection_name_for(transcript)
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        store.delete_collection()
        print(f"Deleted vector store collection: {collection_name}")
    except Exception as e:
        print(f"  Could not delete vector store: {e}")


def get_retriever(vector_store: Chroma, k: int = 4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})