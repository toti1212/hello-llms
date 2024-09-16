import os

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")


def main():
    # parser
    class Food(BaseModel):
        name: str = Field(description="The name of the food")
        ingredients: str = Field(description="The ingredients of the food")
        instructions: str = Field(description="The instructions to make the food")

    parser = JsonOutputParser(pydantic_object=Food)
    format_instructions = parser.get_format_instructions()

    template = """You are a helpful assistant that knows about food. Use the following context to answer the question:
        Context: {context}
        
        Question: {question}
        
        The answer should just be one food recipe.
        If you don't know the answer, just say so. Don't try to make up an answer.
        
        {format_instructions}
        """
    prompt = PromptTemplate(
        input_variables=["context", "question", "format_instructions"],
        template=template,
        partial_variables={"format_instructions": format_instructions},
    )

    # load the data
    print("Loading the data...")
    loader = PyPDFLoader("../docs/data.pdf")
    docs = loader.load_and_split()

    # split the data
    print("Splitting the data...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, chunk_overlap=100, separators=["\n \n", "\n  \n"]
    )
    chunks = text_splitter.split_documents(docs)

    # create the vector db
    print("Creating the vector db...")
    vector_db = FAISS.from_documents(chunks, OpenAIEmbeddings())

    input_query = input("Enter a query: ")
    result = vector_db.similarity_search(input_query)
    # print("Similarity search result:")
    # print(result)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    result = retriever.invoke(input_query)
    # print("Retriever result:")
    # print(result)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    print("RAG chain result:")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    print("RAG chain result:")

    # invoke
    # result = rag_chain.invoke(input_query)
    # print(result)

    # stream
    for s in rag_chain.stream(input_query):
        print(s)

    # batch
    # used for a list of inputs: like: give me a recipie for a cake, a pizza and a sandwich.


if __name__ == "__main__":
    main()
