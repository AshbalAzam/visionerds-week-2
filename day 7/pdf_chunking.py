from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="day 7/test.pdf")
pages = loader.load_and_split()

chunks= len(pages)
print("number of chunks: ", chunks)
# print the whole page
print(pages[4])

# give the first 100 charater
# print(pages[0].page_content[:100]) 

