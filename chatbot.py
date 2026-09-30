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
            text+=page.extract_text()+"\n"
        # st.write(text)  

# chunking
text_splitter=RecursiveCharacterTextSplitter(
    separators=["\n\n","\n"," ","",".  "],
    chunk_size=1000,
    chunk_overlap=200
)
chunks=text_splitter.split_text(text)
st.write(chunks)

#setting up a model for embedding generation
embeddings=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Generation of embeddings and storage in VDB using FAISS
# vector_store=FAISS.from_texts(chunks,embeddings)