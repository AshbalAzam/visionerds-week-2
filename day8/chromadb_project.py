from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

pdf_path = "E:\\visionerds\\Week 2\\day8\\test.pdf"

# 1. Load document
documents = PyPDFLoader(pdf_path).load()

# 2. Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# remove empty chunks
chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

# 3. Embedding model (LangChain wrapper, not raw SentenceTransformer)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store in vector db
vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embedding_model,
    persist_directory="./chroma_database"
)

vector_store.add_documents(documents=chunks)

# 5. Retrieve
query = input("Enter your search query: ")
results = vector_store.similarity_search(
    query, k=5
)

for doc in results:
    print(doc.page_content)