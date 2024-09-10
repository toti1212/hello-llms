import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    
    class Food(BaseModel):
        name: str = Field(description="The name of the food")   
        ingredietns: str = Field(description="The ingredients of the food")

    parser = JsonOutputParser(pydantic_object=Food) 
    parser.get_format_instructions()
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = PromptTemplate(
        input_variables=["input"],
        template="You are a helpful assistant that know about food. You are a chef. {format_instructions} {input}",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = (prompt | llm | parser)
    
    user_input = input("What do you want to eat?")
    response = chain.invoke({"input": user_input})
    
    print(response)
    
    
    
if __name__ == "__main__":
    main()