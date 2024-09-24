import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.chains.llm import LLMChain
load_dotenv()


def main():
    chatbot = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    memory = ConversationBufferMemory(
        chat_memory=FileChatMessageHistory("chat_history.json"),
        memory_key="messages",
        llm=chatbot,
        return_messages=True
    )
    
    # The context is expensive!!!
    # Don't save too much information here 
    prompt = ChatPromptTemplate(
        input_variables=["input", "messages"],
        messages=[
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    chain = LLMChain(llm=chatbot, memory=memory, prompt=prompt)
    while True:
        user_input = input("User: ")
        response = chain({"input": user_input})
        print(f"Assistant: {response}")



if __name__ == "__main__":
    main()
