import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

load_dotenv(Path(__file__).parent / ".env")

DB_DIR = Path(__file__).parent / "db"


def main() -> None:
    if not DB_DIR.exists():
        print("Vector database not found. Run ingest.py first to build the embeddings store.")
        return

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    model = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        verbose=False,
    )

    print("RAG chatbot is ready. Type a question or 'exit' to quit.")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = qa.run(query)
        print(f"Bot: {result}\n")


if __name__ == "__main__":
    main()
