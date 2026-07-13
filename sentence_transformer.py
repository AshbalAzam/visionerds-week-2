from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sample sentences
sentences = [
    "I love programming in Python.",
    "Python is my favorite programming language.",
    "Artificial Intelligence is changing the world.",
    "Machine Learning is a branch of Artificial Intelligence.",
    "I enjoy eating pizza on weekends.",
    "The weather is very hot today."
]

# Generate embeddings
embeding_vector = model.encode(sentences)

# Calculate similarity between all sentences
similarity_matrix = model.similarity(embeding_vector, embeding_vector)

print("Similarity matrix:")

for row in similarity_matrix:
    for score in row:
        print(f"{score:.2f}", end="\t")
    print()

print("\nSentences:")
for i, sentence in enumerate(sentences):
    print(f"{i + 1}. {sentence}")


def most_similar(query, sentences):
    # Convert the query into an embedding
    make_vector = model.encode(query)

    # Compare the query with all sentence embeddings
    scores = model.similarity(make_vector, embeding_vector)

    # Convert tensor to list
    scores = scores.tolist()[0]

    # Find the highest similarity score
    result = scores.index(max(scores))

    return sentences[result], scores[result]


query = input("Enter a query sentence: ")
score = most_similar(query, sentences)

print("\nQuery:")
print(query)

print(f"\nSimilarity Score: {score}")