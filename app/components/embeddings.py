from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import os

logger = get_logger(__name__)

# Pin the cache directory so the model is not re-downloaded on every cold start
HF_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", HF_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", HF_CACHE_DIR)

def get_embedding_model():
    try:
        logger.info("intializing ur huggingface model..")

        model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder = HF_CACHE_DIR
        )
        logger.info("Huggingface embedding model load ✔️✔️")
        return model
    except Exception as e:
        error_message = CustomException("Failed to load embedding model",e)
        logger.error(str(error_message))
        raise  error_message