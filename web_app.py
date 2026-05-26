import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from openai import OpenAIError, RateLimitError

load_dotenv(Path(__file__).parent / ".env")

SOURCE_DIR = Path(__file__).parent / "source_documents"
PERSIST_DIR = Path(__file__).parent / "db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

st.set_page_config(page_title="Secure Corporate PDF RAG Chatbot", layout="wide")

st.title("Secure Corporate PDF RAG Chatbot")
st.write(
    "Ask questions from your source documents using a vector database backed by OpenAI embeddings. "
    "If the index is missing or outdated, click `Build vector DB` first."
)


def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")


def has_openai_key() -> bool:
    key = get_openai_key().strip()
    return bool(key) and key != "your_openai_api_key_here"


def load_documents(source_dir: Path):
    documents = []
    if not source_dir.exists():
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
            continue

        documents.extend(loader.load())

    return documents


def build_vector_db(source_dir: Path, persist_dir: Path):
    documents = load_documents(source_dir)
    if not documents:
        st.error("No documents found in source_documents/. Add .txt, .md, or .pdf files and rebuild.")
        return None

    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()

    try:
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
        vectordb.persist()
        return vectordb
    except RateLimitError as exc:
        st.error(
            "OpenAI quota exceeded while building the vector DB. "
            "Check your OpenAI plan and billing details, or use a different API key."
        )
        st.write(str(exc))
        return None
    except OpenAIError as exc:
        st.error("OpenAI error while building the vector DB.")
        st.write(str(exc))
        return None
    except Exception as exc:
        st.error("Unexpected error while building the vector DB.")
        st.write(str(exc))
        return None


def load_vector_db(persist_dir: Path):
    if not persist_dir.exists():
        return None

    embeddings = OpenAIEmbeddings()
    try:
        vectordb = Chroma(persist_directory=str(persist_dir), embedding=embeddings)
    except Exception:
        return None
    return vectordb


def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    model = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        verbose=False,
    )


with st.sidebar:
    st.header("Vector Database")
    st.write(f"Source docs: `{SOURCE_DIR}`")
    st.write(f"Persist directory: `{PERSIST_DIR}`")
    if has_openai_key():
        st.success("OpenAI API key loaded.")
        st.write("You can build the vector DB now.")
    else:
        st.warning("OPENAI_API_KEY is missing.")
        st.markdown(
            "1. Get an API key: [OpenAI API keys](https://platform.openai.com/account/api-keys)  \
"
            "2. Add it to `rag_chatbot/.env` as `OPENAI_API_KEY=your_key_here`  \
"
            "3. Restart this app if needed."
        )

    if st.button("Build vector DB"):
        if not has_openai_key():
            st.error("Cannot build vector DB because OPENAI_API_KEY is missing.")
        else:
            vectordb = build_vector_db(SOURCE_DIR, PERSIST_DIR)
            if vectordb:
                st.success("Vector DB built successfully.")

    if st.button("Reload vector DB"):
        if PERSIST_DIR.exists():
            st.success("Reloaded vector DB from disk.")
        else:
            st.error("No vector DB found. Build it first.")


st.subheader("Source document preview")
uploaded_files = st.file_uploader(
    "Upload PDF, text, or markdown files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"### {uploaded_file.name}")
        try:
            file_bytes = uploaded_file.read()
            file_text = file_bytes.decode("utf-8")
        except Exception:
            st.write("Preview not available for this file type.")
        else:
            st.text_area("File preview", file_text, height=300, key=uploaded_file.name)
    st.success("Upload complete.")
else:
    st.info("You can upload documents here for a quick preview.")

st.subheader("Ask a question")
query = st.text_input("Enter your question here")
if st.button("Get answer"):
    if not query:
        st.warning("Please enter a question first.")
    elif not has_openai_key():
        st.error("Set OPENAI_API_KEY in .env before using the answer feature.")
    else:
        vectordb = load_vector_db(PERSIST_DIR)
        if vectordb is None:
            st.error("No vector DB found. Click Build vector DB first.")
        else:
            qa = create_qa_chain(vectordb)
            try:
                with st.spinner("Searching documents and generating an answer..."):
                    answer = qa.run(query)
                st.write("**Answer:**")
                st.write(answer)
            except RateLimitError as exc:
                st.error(
                    "OpenAI quota exceeded while generating an answer. "
                    "Check your plan and billing details."
                )
                st.write(str(exc))
            except OpenAIError as exc:
                st.error("OpenAI error while generating an answer.")
                st.write(str(exc))
            except Exception as exc:
                st.error("Unexpected error while generating an answer.")
                st.write(str(exc))
