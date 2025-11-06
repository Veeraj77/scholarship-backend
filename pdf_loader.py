import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdfs():
    """Loads text from all PDFs in ../DATA/PDFS"""
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA", "PDFS")
    print(f"📂 Looking for PDFs in: {pdf_dir}")

    texts = []
    if not os.path.exists(pdf_dir):
        print(f"❌ ERROR: Folder not found: {pdf_dir}")
        return texts

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ No PDF files found in {pdf_dir}")
        return texts

    for file in pdf_files:
        path = os.path.join(pdf_dir, file)
        print(f"➡️ Reading {file}...")
        try:
            with pdfplumber.open(path) as pdf:
                content = "\n".join([p.extract_text() or "" for p in pdf.pages])
                if content.strip():
                    texts.append({"filename": file, "content": content})
                    print(f"✅ Loaded {file} ({len(content)} chars)")
        except Exception as e:
            print(f"⚠️ Failed to load {file}: {e}")
    return texts

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks"""
    from langchain.schema import Document
    print("\n✂️ Splitting documents...")
    langchain_docs = [
        Document(page_content=d["content"], metadata={"source": d["filename"]}) for d in docs
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(langchain_docs)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks

def create_and_save_vector_store(chunks, db_path="faiss_index"):
    """Create FAISS index"""
    print("\n🧠 Creating FAISS vectorstore...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    vectorstore.save_local(db_path)
    print(f"✅ Saved FAISS index to {db_path}")
    return vectorstore

if __name__ == "__main__":
    docs = load_pdfs()
    if docs:
        chunks = split_documents(docs)
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index")
        vectorstore = create_and_save_vector_store(chunks, db_path)
        print("\n🎉 FAISS index ready for ScholarBot!")
    else:
        print("⚠️ No documents to process.")
