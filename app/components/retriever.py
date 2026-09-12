from app.components.llm import load_llm
from app.components.vector_store import load_vector_store
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = get_logger(__name__)

PROMPT = ChatPromptTemplate.from_template("""
Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context:
{context}

Question:
{input}

Answer:
""")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain():
    try:
        logger.info("Loading vectorstore for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vectorstore not present or empty")

        llm = load_llm()

        if llm is None:
            raise CustomException("LLM not loaded...")

        retriever = db.as_retriever(search_kwargs={'k': 1})

        # ✅ Direct LCEL chain — no create_retrieval_chain wrapper
        # Input: plain string (user question)
        # Output: plain string (answer)
        qa_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | PROMPT
            | llm
            | StrOutputParser()
        )

        logger.info("Successfully created the QA chain")
        return qa_chain
    except Exception as e:
        error_message = CustomException("failed to make QA chain", e)
        logger.error(str(error_message))
        return None