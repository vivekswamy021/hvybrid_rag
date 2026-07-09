import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

st.set_page_config(page_title="Structure-Aware Hybrid RAG", page_icon="🔍", layout="wide")
st.title("🔍 Advanced Layout-Aware Hybrid RAG")
st.subheader("Process PDFs with multi-columns, text, and complex tables.")

# 1. Fetch the Groq API key securely from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Please configure your GROQ_API_KEY in the Streamlit Secrets settings.")
    st.stop()

# 2. Cached initialization functions to optimize memory and speed
@st.cache_resource
def load_embedding_and_reranker():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    return embedding_model, reranker

embedding_model, reranker = load_embedding_and_reranker()

# 3. Document Processing Pipeline
def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.spinner("Analyzing document structure & extracting elements (Tables/Columns)..."):
        loader = UnstructuredPDFLoader(file_path=tmp_path, strategy="hi_res", infer_table_structure=True)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
    os.unlink(tmp_path) # Clean up temp file
    return chunks

# --- UI Layout ---
uploaded_file = st.file_uploader("Upload your document (PDF)", type=["pdf"])

if uploaded_file:
    # Initialize session state for the retrievers
    if "hybrid_retriever" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        chunks = process_uploaded_file(uploaded_file)
        
        # Build Vector DB (In-Memory Chroma instance for ephemeral Streamlit apps)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding_model)
        semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # Build BM25
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 5
        
        # Ensemble Combine
        st.session_state.hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5]
        )
        st.session_state.file_name = uploaded_file.name
        st.success("Document ingestion complete!")

    # Question Answering UI
    user_query = st.text_input("Ask a question about tables or text layout in this document:")
    
    if user_query:
        with st.spinner("Searching and generating answer..."):
            # Step 1: Hybrid Retrieve
            candidate_docs = st.session_state.hybrid_retriever.invoke(user_query)
            
            # Step 2: Cross-Encoder Rerank
            pairs = [[user_query, doc.page_content] for doc in candidate_docs]
            scores = reranker.predict(pairs)
            scored_docs = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
            final_docs = [doc for score, doc in scored_docs[:3]]
            
            # Step 3: LLM Synthesize
            context_str = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
            system_prompt = (
                "You are an expert document analyst. Use the provided context fragments to answer the user question.\n"
                "The context might contain structured tables or multi-column data layouts.\n"
                f"Context:\n{context_str}"
            )
            
            llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.1)
            response = llm.invoke([("system", system_prompt), ("human", user_query)])
            
            st.markdown("### ✨ Answer")
            st.write(response.content)
            
            with st.expander("🔍 View Retrieved Context Fragments"):
                for idx, doc in enumerate(final_docs):
                    st.markdown(f"**Fragment {idx+1}:**")
                    st.code(doc.page_content, language="markdown")
