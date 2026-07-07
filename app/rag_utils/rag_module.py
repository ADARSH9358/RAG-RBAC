# ==========================================
# ============== 1. CONFIG =================
# ==========================================
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import chromadb
import pandas as pd
from langchain.retrievers import ContextualCompressionRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_compressors import CohereRerank
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# Set tracing environment variables immediately
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "RAG"

# Initialize your API Key first so it's globally available below
openapi_key = os.environ.get("GROQ_API_KEY", "")

# ==========================================
# ====== 2. EMBEDDINGS & VECTORSTORE =======
# ==========================================
huggingface_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

chroma_client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CROMA_API_KEY"),   # Note: matching your existing typo configuration (CROMA)
)

vectorstore = Chroma(
    client=chroma_client,
    collection_name="my_collection",
    embedding_function=huggingface_embeddings,
)


# ==========================================
# =========== 3. UTILITY METHODS ===========
# ==========================================
def _is_quota_exceeded_error(err: Exception) -> bool:
    message = str(err).lower()
    return "quota exceeded" in message and "upsert" in message


def embed_documents_to_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore.add_documents(splits)
    
    print("Documents embedded and saved to vectorstore.")
    print("Total documents:", len(vectorstore.get()["documents"]))


def load_file(filepath, role):
    ext = Path(filepath).suffix.lower()
    try:
        if ext == ".csv":
            df1 = pd.read_csv(filepath)
            documents = []
            for row in df1.to_dict(orient="records"):
                content = "\n".join(f"{k}: {v}" for k, v in row.items())
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"role": role.lower(), "source": Path(filepath).name}
                    )
                )
            return documents

        elif ext == ".md":
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return [
                Document(
                    page_content=content,
                    metadata={"role": role.lower(), "source": Path(filepath).name}
                )
            ]
        else:
            return None

    except Exception as e:
        print(f"Failed to process {filepath}: {e}")
        return None


def run_indexer():
    from .turso_client import connect as turso_connect
    conn = turso_connect()
    c = conn.cursor()
    c.execute("SELECT id, filepath, role FROM documents WHERE embedded = 0")
    pending_rows = c.fetchall()

    all_docs = []
    embedded_doc_ids = []

    try:
        for doc_id, path, role in pending_rows:
            docs = load_file(path, role)
            if not docs:
                continue

            if isinstance(docs, list):
                all_docs.extend(docs)
            else:
                all_docs.append(docs)
            embedded_doc_ids.append(doc_id)

        if not all_docs:
            return {
                "indexed_chunks": 0,
                "indexed_docs": 0,
                "pending_docs": len(pending_rows),
                "quota_exceeded": False,
            }

        embed_documents_to_vectorstore(all_docs)

        for doc_id in embedded_doc_ids:
            c.execute("UPDATE documents SET embedded = 1 WHERE id = ?", (doc_id,))

        conn.commit()
        print(f"Indexed {len(all_docs)} document chunks.")
        return {
            "indexed_chunks": len(all_docs),
            "indexed_docs": len(embedded_doc_ids),
            "pending_docs": max(len(pending_rows) - len(embedded_doc_ids), 0),
            "quota_exceeded": False,
        }

    except Exception as e:
        if _is_quota_exceeded_error(e):
            conn.rollback()
            print(f"⚠️ Index skipped due to Chroma quota: {e}")
            return {
                "indexed_chunks": 0,
                "indexed_docs": 0,
                "pending_docs": len(pending_rows),
                "quota_exceeded": True,
                "error": str(e),
            }
        raise
    finally:
        conn.close()


# ==========================================
# ========== 4. PROMPT TEMPLATE ===========
# ==========================================
system_prompt = (
    "You are an assistant for summarizing and answering queries from internal company documents.\n"
    "Always use the retrieved context to answer the query, even if partial.\n"
    "Do not guess. If data is not found, explain what you searched for.\n"
    "When responding:\n"
    "- Add **Source** from document metadata if possible.\n"
    "- Use headers\n"
    "- Use bullet points\n"
    "- For CSV-style data, format in table with two columns\n"
    "\n{context}"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


# ==========================================
# ============ 5. LLM MODEL ================
# ==========================================
model = ChatOpenAI(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    api_key=openapi_key,
    base_url="https://api.groq.com/openai/v1",
    temperature=0.2
)

question_answering_chain = create_stuff_documents_chain(model, chat_prompt)


# ==========================================
# ============ 6. RERANK & CHAIN ===========
# ==========================================
def wrap_with_reranker(retriever, cohere_api_key, top_n=4):
    reranker = CohereRerank(cohere_api_key=cohere_api_key, top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever
    )


def get_rag_chain(user_role: str, cohere_api_key: str = None):
    user_role = user_role.lower()

    if user_role == "c-level":
        # C-level sees everything
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    elif user_role == "general":
        # General role sees only general documents
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 4,
            "filter": {"role": "general"}
        })
    else:
        # All other roles see their docs + general
        retriever = vectorstore.as_retriever(search_kwargs={
            "k": 4,
            "filter": {
                "role": {"$in": [user_role, "general"]}
            }
        })

    # Wrap with reranker if key is present
    if cohere_api_key:
        print("Using cohere reranker")
        retriever = wrap_with_reranker(retriever, cohere_api_key)

    return create_retrieval_chain(retriever, question_answering_chain)