import hashlib
import os
import tempfile
from pathlib import Path
from typing import Iterable

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

APP_TITLE = "Document RAG Assistant"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


st.set_page_config(page_title=APP_TITLE, page_icon="doc", layout="wide")

st.markdown(
    """
    <style>
        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
        }
        [data-testid="stSidebar"] {
            background: #121826;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: #f8fafc !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #334155;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: #0f172a;
            border-color: #475569;
        }
        [data-testid="stFileUploaderDropzone"] svg {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] [data-baseweb="slider"] div {
            color: #f8fafc;
        }
        .source-box {
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 0.8rem;
            background: #111827;
            color: #f8fafc;
            margin-bottom: 0.75rem;
            font-size: 0.92rem;
        }
        .source-title {
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


def _load_pdf(uploaded_file) -> list[Document]:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        documents = PyPDFLoader(temp_path).load()
    finally:
        Path(temp_path).unlink(missing_ok=True)

    for document in documents:
        document.metadata["source_file"] = uploaded_file.name
        document.metadata["file_type"] = "pdf"
    return documents


def _load_text(uploaded_file) -> list[Document]:
    raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    return [
        Document(
            page_content=raw_text,
            metadata={"source_file": uploaded_file.name, "file_type": "text"},
        )
    ]


def load_uploaded_documents(uploaded_files: Iterable) -> list[Document]:
    documents: list[Document] = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            documents.extend(_load_pdf(uploaded_file))
        elif file_name.endswith((".txt", ".md")):
            documents.extend(_load_text(uploaded_file))
    return documents


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


@st.cache_resource(show_spinner=False)
def build_vector_store(file_signatures: tuple, chunk_size: int, chunk_overlap: int):
    uploaded_files = st.session_state.get("uploaded_files", [])
    documents = load_uploaded_documents(uploaded_files)
    if not documents:
        return None, []

    chunks = split_documents(documents, chunk_size, chunk_overlap)
    collection_hash = hashlib.sha1(str(file_signatures).encode("utf-8")).hexdigest()[:12]
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=f"uploaded-rag-{collection_hash}",
    )
    return vector_store, chunks


def make_file_signatures(uploaded_files) -> tuple:
    return tuple((file.name, file.size, file.type) for file in uploaded_files)


def format_source(document: Document, index: int) -> str:
    source = document.metadata.get("source_file", "Uploaded document")
    page = document.metadata.get("page")
    page_text = f" · page {int(page) + 1}" if page is not None else ""
    snippet = document.page_content.strip().replace("\n", " ")
    snippet = " ".join(snippet.split())[:700]
    return f"""
        <div class="source-box">
            <div class="source-title">{index}. {source}{page_text}</div>
            <div>{snippet}</div>
        </div>
    """


def answer_with_context(vector_store: Chroma, question: str, top_k: int) -> dict:
    source_documents = vector_store.similarity_search(question, k=top_k)
    context = "\n\n".join(
        f"Source {index}: {document.page_content}"
        for index, document in enumerate(source_documents, start=1)
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful RAG assistant. Answer only from the supplied context. "
                "If the context does not contain the answer, say that you could not find it "
                "in the uploaded document. Keep the answer clear and concise.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )

    messages = prompt.format_messages(context=context, input=question)
    response = get_llm().invoke(messages)
    return {"answer": response.content, "context": source_documents}


with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    st.session_state["uploaded_files"] = uploaded_files or []

    st.divider()
    st.header("Retrieval")
    top_k = st.slider("Context chunks", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider(
        "Chunk overlap", min_value=0, max_value=500, value=200, step=50
    )

    st.divider()
    if os.getenv("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY found")
    else:
        st.warning("Add OPENAI_API_KEY to .env to generate answers")


st.title(APP_TITLE)
st.caption("Upload PDFs or text files, build a vector index, and ask questions grounded in your documents.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not uploaded_files:
    st.info("Upload one or more PDF, TXT, or Markdown files from the sidebar to start.")
    st.stop()

file_signatures = make_file_signatures(uploaded_files)

with st.spinner("Reading files and building the vector index..."):
    vector_store, chunks = build_vector_store(file_signatures, chunk_size, chunk_overlap)

if vector_store is None:
    st.error("I could not read any supported content from the uploaded files.")
    st.stop()

st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the uploaded documents")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is missing. Add it to `.env`, then restart Streamlit.")
            st.stop()

        with st.spinner("Searching the document and writing the answer..."):
            result = answer_with_context(vector_store, question, top_k)
            answer = result.get("answer", "I could not generate an answer.")
            sources = result.get("context", [])

        st.markdown(answer)
        with st.expander("Sources", expanded=True):
            for index, source_doc in enumerate(sources, start=1):
                st.markdown(format_source(source_doc, index), unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer})
