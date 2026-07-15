from langchain_text_splitters import RecursiveCharacterTextSplitter

sample= '''Topic 2: Chunking
What it covers: breaking a long document into pieces small enough to embed and search properly, instead
of trying to embed the whole thing at once.
What it means: you can't just take a 50-page PDF, turn it into one embedding, and expect to answer specific
questions about page 30. Embeddings work best on chunks of a reasonable size, maybe a few hundred words
each. The simplest approach is fixed-size chunking, just cut the text every N characters or words, sometimes
with a bit of overlap between chunks so you don't cut a sentence in half and lose context. A better approach is
semantic chunking, splitting on natural boundaries like paragraphs or sections instead of a raw character
count. Start with fixed-size, it's good enough to learn the concept, semantic chunking is the "something
extra" here.
What to google:
"text chunking for rag"
"chunk overlap explained"
"recursive character text splitter"
What to build:
take any PDF or long text file (a manual, an article, whatever you have lying around) and write a function that
splits it into chunks of roughly 300-500 words, with maybe 50 words of overlap between consecutive chunks.
Print out the first few chunks and manually check they make sense, that you're not cutting off mid-sentence in a
way that loses the point.
Try changing your chunk size smaller and bigger, and just notice how the chunks read differently. You don't need
to write this up anywhere, just build a feel for it.'''


text_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=5)
texts = text_splitter.split_text(sample)

length= len(texts)
print(f"Number of chunks: ", length)

print(texts)