from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

def analyze_resume(vectorstore, jd_text: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    You are an expert technical recruiter.

    Analyze how well the resume matches the job description using the provided context.

    Job Description:
    {input}

    Relevant Resume Context:
    {context}

    Return the result in the following structured format (NOT JSON):

    Match Score: <number between 0 and 100>

    Strong Skills:
    Write a clear paragraph describing the candidate’s strongest matching skills.

    Missing Skills:
    Write a paragraph explaining the important skills or areas missing or weak.

    Suggestions:
    Write a paragraph with actionable improvements for the candidate’s resume.

    Recommendation:
    Write a final paragraph with your hiring recommendation and reasoning.

    IMPORTANT:
    - Do NOT return JSON
    - Do NOT use bullet points
    - Use clear paragraphs under each heading
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": jd_text})
    return response["answer"]
