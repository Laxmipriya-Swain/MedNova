from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

logger = get_logger(__name__)

# ✅ ChatPromptTemplate required for ChatGroq (chat model)
PROMPT = ChatPromptTemplate.from_template("""
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{input}

Answer:
""")


def create_qa_chain():
    try:
        logger.info("Loading vectorstore for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vectorstore not present or empty")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded...")

        # ✅ LCEL chain compatible with langchain 0.3.x + ChatGroq
        combine_chain = create_stuff_documents_chain(llm, PROMPT)
        qa_chain = create_retrieval_chain(
            db.as_retriever(search_kwargs={'k': 1}),
            combine_chain
        )

        logger.info("Successfully created the QA chain")
        return qa_chain
    except Exception as e:
        error_message = CustomException("failed to make QA chain", e)
        logger.error(str(error_message))
        return None