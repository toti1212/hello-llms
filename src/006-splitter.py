import os

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
_  = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    loader = PyPDFLoader("../docs/data.pdf")
    
    docs = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n \n", "\n  \n"]
    )
    
    chunks = text_splitter.split_documents(docs[2:])
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} - Length: {len(chunk.page_content)}")
    


if __name__ == "__main__":
    main()

