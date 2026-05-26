import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

SOURCE_DIR = Path(__file__).parent / "source_documents"
PERSIST_DIR = Path(__file__).parent / "db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents(source_dir: Path) -> List[Document]:
    documents: List[Document] = []
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        return documents

    for path in sorted(source_dir.iterdir()):
        if path.is_dir():
            continue

        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            print(f"Skipping unsupported file type: {path.name}")
            continue

        print(f"Loading: {path.name}")
        documents.extend(loader.load())

    return documents


def main() -> None:
    documents = load_documents(SOURCE_DIR)
    if not documents:
        print("No documents found in source_documents/. Add .txt, .md, or .pdf files and run again.")
        return

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
    )
    vectordb.persist()

    print(f"Indexed {len(chunks)} chunks into Chroma at: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
