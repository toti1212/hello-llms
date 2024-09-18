import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
load_dotenv()

def main():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


    # I thought that this works, but not --------
    # template = """You are a helpful assistant.
    # Chat history: {chat_history}
    # Human: {input}  
    # AI:"""
    
    # prompt = PromptTemplate(
    #     input_variables=["chat_history", "input"],
    #     template=template
    # )
    # -------------------------------------------
    
    # Correct way to do it ----------------------------
    prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ]
    )
    # -------------------------------------------   
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    while True:
        user_input = input("You: ")
        response = chain({"user_input": user_input})
        print("Assistant: ", response["text"])


if __name__ == "__main__":
    main()
