import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever


def get_llm():
    return ChatOllama(model="llama3.2", temperature=0.3)


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def _build_chain(vector_store):
    retriever = get_retriever(vector_store, k=4)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert content assistant. Answer the user's question
based ONLY on the transcript context provided below.

If the answer is not found in the context, say:
"I could not find this information in the transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from transcript:
{context}"""
        ),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str):
    vector_store = build_vector_store(transcript)
    return _build_chain(vector_store)


def load_rag_chain(transcript: str):        # ← changed: now takes transcript
    vector_store = load_vector_store(transcript)
    return _build_chain(vector_store)


def ask_question(rag_chain, question: str) -> str:
    print(f"Question : {question}")
    answer = rag_chain.invoke(question)
    print(f"answer : {answer}")
    return answer