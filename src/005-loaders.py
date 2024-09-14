from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader("../docs/data.pdf")
docs = loader.load()

print(docs[30].page_content)