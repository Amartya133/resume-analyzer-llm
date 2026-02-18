from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def chat_with_resume(vectorstore, question: str, chat_history: list, analysis: str, jd: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent career assistant.

    You have access to:
    1. Resume content
    2. Resume analysis report
    3. Job description

    Use all sources to answer the user’s question.

    If the user asks about skills gaps, improvements, or job fit, compare resume with job description.

    Chat History:
    {chat_history}

    ----------------------

    Job Description:
    {jd}

    ----------------------

    Resume Analysis Report:
    {analysis}

    ----------------------

    Resume Context:
    {context}

    ----------------------

    User Question:
    {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({
        "input": question,
        "chat_history": "\n".join(chat_history),
        "analysis": analysis,
        "jd": jd
    })

    return response["answer"]