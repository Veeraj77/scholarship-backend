import os
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np

# --- 1. IMPORT THE BASE LIBRARIES ---
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss

# --- 2. IMPORT THE *ONE* CLASS NEEDED TO READ YOUR .pkl FILE ---
# This is ONLY here so pickle.load() can understand your index.pkl file.
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


# --- ENV SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("❌ ERROR: GOOGLE_API_KEY environment variable is NOT set.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

# --- FASTAPI APP ---
app = FastAPI(title="ScholarBot RAG API (No LangChain - Final)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# --- GLOBAL MODELS (Load only once) ---
llm = None
embedding_model = None
faiss_index = None
docstore = None
index_to_docstore_id = None

@app.on_event("startup")
async def startup_event():
    global llm, embedding_model, faiss_index, docstore, index_to_docstore_id
    
    try:
        # --- 1. Load Chat Model (Gemini) ---
        print("🧠 Initializing Gemini-2.5-Flash...")
        llm = genai.GenerativeModel("gemini-2.5-flash")
        print("✅ Gemini-2.5-Flash initialized.")

        # --- 2. Load Embedding Model (Free, Local) ---
        print("🤖 Loading free SentenceTransformer embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Free embedding model loaded.")

        # --- 3. Load FAISS Vector Database Manually ---
        # --- TYPO FIXED HERE ---
        db_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        faiss_file = os.path.join(db_path, "index.faiss")
        pkl_file = os.path.join(db_path, "index.pkl")
        
        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
            print(f"❌ ERROR: FAISS index files not found in {db_path}")
            print("➡️ Please run your `build_index.py` script first.")
            return

        print(f"📂 Loading FAISS index from {db_path}...")
        faiss_index = faiss.read_index(faiss_file)
        
        with open(pkl_file, "rb") as f:
            # This loads the two parts langchain saved: the document texts and the ID mapping
            docstore, index_to_docstore_id = pickle.load(f)
            
        print("✅ FAISS index and docstore loaded.")
        print("🚀 ScholarBot is ready!")
        
    except Exception as e:
        print(f"❌ CRITICAL STARTUP ERROR: {e}")

def get_rag_response(query: str) -> str:
    """
    Manually performs the Retrieval-Augmented Generation (RAG) process.
    """
    if not all([llm, embedding_model, faiss_index, docstore, index_to_docstore_id]):
        raise Exception("Models not initialized. Check startup logs.")

    # 1. Embed the query
    print(f"Embedding query: '{query}'")
    query_embedding = embedding_model.encode([query])

    # 2. Search FAISS for relevant documents
    k = 3 # Get top 3 results
    D, I = faiss_index.search(np.array(query_embedding, dtype=np.float32), k)

    # 3. Get the document text from the docstore
    retrieved_docs = []
    for i in I[0]:
        if i == -1: # -1 means no result
            continue
        doc_id = index_to_docstore_id[i]
        retrieved_docs.append(docstore.search(doc_id))

    # --- THIS IS THE FINAL, SMARTEST PROMPT LOGIC ---

    if not retrieved_docs:
        print("⚠️ No relevant documents found. Using general knowledge.")
        # Context is empty, so we give it a "general conversation" prompt
        prompt_template = f"""
        You are a friendly and helpful scholarship assistant named ScholarBot.
        The user said: "{query}"
        Respond in a friendly, conversational way.
        """
    else:
        # Context was found, so we use the RAG prompt
        print(f"✅ Found {len(retrieved_docs)} relevant documents.")
        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        
        prompt_template = f"""
        You are an expert scholarship assistant. Your primary goal is to answer the user's "Question".
        You will be given "Context" from a document database to help you.

        **CRITICAL INSTRUCTIONS:**
        1.  First, analyze the user's "Question".
        2.  Next, analyze the "Context". Ask yourself: "Does this context *actually* answer the user's question?"
        3.  **If the Context IS NOT RELEVANT:**
            * For example, if the Question is "What are the names of scholarships for boys?" and the Context is a list of "Engineering Majors" or a rule about "Class I to X".
            * In this case, you **MUST IGNORE THE CONTEXT** and answer the "Question" using your own general knowledge.
        4.  **If the Context IS RELEVANT:**
            * Use the information from the "Context" to provide a specific answer.
        
        **Context:**
        {context}

        **Question:**
        {query}

        **Answer:**
        """
    # --- END OF NEW LOGIC ---

    # 4. Call the LLM
    print("🤖 Calling Gemini...")
    response = llm.generate_content(prompt_template)
    
    return response.text

@app.get("/")
def root():
    return {"message": "ScholarBot RAG API (No LangChain - Final) running."}

@app.post("/query")
async def query_bot(request: QueryRequest):
    if not llm:
        return JSONResponse(status_code=500, content={"error": "Models not initialized."})
    
    try:
        answer = get_rag_response(request.query)
        return {"answer": answer}
    except Exception as e:
        print(f"❌ Query error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})