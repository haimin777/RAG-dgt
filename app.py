import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv

load_dotenv()

# ================== MULTILINGUAL CONFIG (CRUCIAL FOR SPANISH) ==================
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")  # Best for Spanish + English

# LLM choice (set LLM_PROVIDER=grok or LLM_PROVIDER=ollama)
llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()

if llm_provider == "grok":
    # Grok API via OpenAI-compatible endpoint
    # Expected env vars:
    #   XAI_API_KEY or OPENAI_API_KEY (required)
    #   GROK_MODEL (optional, default grok-4-0709)
    #   GROK_CONTEXT_WINDOW (optional, default 128000)
    from llama_index.llms.openai_like import OpenAILike

    grok_api_key = os.getenv("XAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not grok_api_key:
        raise RuntimeError("Missing XAI_API_KEY for Grok. Set XAI_API_KEY in your environment.")

    grok_model = os.getenv("GROK_MODEL", "grok-4-0709")
    grok_context_window = int(os.getenv("GROK_CONTEXT_WINDOW", "128000"))

    Settings.llm = OpenAILike(
        model=grok_model,
        api_base="https://api.x.ai/v1",
        api_key=grok_api_key,
        context_window=grok_context_window,
        is_chat_model=True,
        is_function_calling_model=False,
        timeout=180.0,
    )
else:
    Settings.llm = Ollama(model="llama3.2", request_timeout=180.0)  # Works great in Spanish

# ================== BUILD / LOAD ==================
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    print("Construyendo índice...")
    documents = SimpleDirectoryReader("driving_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=6)

# Simple chat
while True:
    q = input("\nPregunta sobre el examen teórico (o 'salir'): ")
    if q.lower() in ["salir", "quit", "exit"]: break
    response = query_engine.query(q)
    print("\nRespuesta:", response)
