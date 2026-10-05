# # testing
# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# api_key = os.getenv("GEMINI_API_KEY")

# client = OpenAI(
#     api_key=api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# response = client.chat.completions.create(
#     model="gemini-flash-lite-latest",
#     messages=[
#         {"role": "user", "content": "What is Generative AI? Explain in one sentence."}
#     ]
# )

# print(response.choices[0].message.content)

# Creating Simple UI
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough 
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print("API key loaded:", GEMINI_API_KEY is not None)
st.header("QueryForge")
with st.sidebar:
    st.title("Your documents")
    file= st.file_uploader("Upload the file here to get the summary",type="pdf")

#Extracting and chunking the contents from pdf
if file is not None:
    #extracting text
    with pdfplumber.open(file) as pdf:
        text=""
        for page in pdf.pages:
            page_text=page.extract_text()
            if page_text:
                text += page_text + "\n"
        # st.write(text)  

# chunking
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n"," ","",".  "],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_text(text)
    # st.write(chunks)

#setting up a model for embedding generation
    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

#Generation of embeddings and storage in VDB using FAISS
    vector_store=FAISS.from_texts(chunks,embeddings)

#taking user question
    user_question=st.text_input("Type your question here")

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

#similarity search
    retriever=vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":4} #return k closest match
    )

#define the llm
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature = 0.5, #randomness of the model
        max_output_tokens=1000,
        google_api_key=GEMINI_API_KEY
    )

#provide system prompts
    prompt=ChatPromptTemplate.from_messages([
        ("system","You are a helpful assistant answering the questions about the PDF document.\n\n"
        "Guidelines:\n"
        "1.Provide complete,well-explained answer using the context given below.\n"
        "2.Include relevant details,numbers, and explainations to give a thorough response.\n"
        "3.If the context mentions related information,include it to give a fuller picture.\n"
        "4.Only use the information given in the context - do not use outside knowledge.\n"
        "5.Summarize long information,ideally in bullets wherever needed.\n"
        "6,If the information is not there, say so politely\n\n"
        "Context:\n{context}"),
        ("human","{question}")
    ])

    chain = (
        {"context":retriever | format_docs,"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    if user_question:
        response=chain.invoke(user_question)
        st.write(response)