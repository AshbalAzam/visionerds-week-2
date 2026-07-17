from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


loader = PyPDFLoader(file_path="./day9/test.pdf")
documents = loader.load()

text_split = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)

chunks = text_split.split_documents(documents)
texts = [doc.page_content for doc in chunks]

# print(texts)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
doc_embeddings = embedding_model.embed_documents(texts=texts)

# print(doc_embeddings)


vector_store = Chroma(
    collection_name="RAG_v1",
    embedding_function=embedding_model,
    persist_directory="./chroma_database"
)

vector_store.add_documents(chunks)

query= input("enter your query ")
result = vector_store.similarity_search(query,k=4)
# for i, doc in enumerate(result, start=1):
#     print(i,"-", doc.page_content)

load_dotenv()

system_prompt="give the answer according to the context if the user input is some how similar with context give answer and the topic is totally out from the topic just say i dont know, and the user input is {query} and the context is".join([doc.page_content for doc in result])

open_router_api_key = os.getenv("OPENROUTER_API_KEY")
openrouter_url = os.getenv("OPENROUTER_BASE_URL")
model = os.getenv("OPENROUTER_MODEL")

client = OpenAI(api_key=open_router_api_key, base_url=openrouter_url)

chat = client.chat.completions.create(model=model, 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.7,)

reply = chat.choices[0].message.content
print(reply,"\n\n\n\n")

# for i, doc in enumerate(result, start=1):
#     print(i,"-", doc.page_content)