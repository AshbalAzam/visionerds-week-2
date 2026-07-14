from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

loader = PyPDFLoader(file_path="day 7/test.pdf")
pages = loader.load_and_split()

pages_text = [page.page_content for page in pages]

# Generate embeddings
embedding_vector = model.encode(pages_text)

similarity_matrix = model.similarity(embedding_vector, embedding_vector)

def most_similar(query, pages_text):
    make_vector = model.encode(query)

    # Compare the query with all sentence embeddings
    scores = model.similarity(make_vector, embedding_vector)

    # Convert tensor to list
    scores = scores.tolist()[0]

    result = scores.index(max(scores))
    return result


query = input("Enter a query sentence: ")
score = most_similar(query, pages_text)

print("\nQuery:")
print(query)
print("Similarity Score:", score, pages_text[score])