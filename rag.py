import os

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv

load_dotenv()


def configure_llm() -> None:
    # ================== EMBEDDINGS ==================
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI embeddings.")
    embed_model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    Settings.embed_model = OpenAIEmbedding(
        model=embed_model,
        api_key=openai_api_key,
    )

    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if llm_provider == "grok":
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
    elif llm_provider == "groq":
        from llama_index.llms.openai_like import OpenAILike

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("Missing GROQ_API_KEY for Groq.")

        groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        Settings.llm = OpenAILike(
            model=groq_model,
            api_base="https://api.groq.com/openai/v1",
            api_key=groq_api_key,
            is_chat_model=True,
            is_function_calling_model=False,
            timeout=180.0,
        )
    else:
        Settings.llm = Ollama(model="llama3.2", request_timeout=180.0)


def get_query_engine(persist_dir: str = "./storage", data_dir: str = "driving_data"):
    configure_llm()
    prebuilt_only = os.getenv("PREBUILT_INDEX", "1") == "1"
    if not os.path.exists(persist_dir):
        if prebuilt_only:
            raise RuntimeError(
                f"Prebuilt index not found at {persist_dir}. "
                "Build it locally and include the storage directory in the image."
            )
        documents = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)

    top_k = int(os.getenv("RAG_TOP_K", "2"))
    response_mode = os.getenv("RAG_RESPONSE_MODE", "compact")
    return index.as_query_engine(similarity_top_k=top_k, response_mode=response_mode)
