import os
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    # Few shot examples
    # Here we provide the model with examples of input and output
    examples = [
        {"input": "Es mejor que me vaya", "output": "Me re fui"},
        {"input": "Me gusta demasiado", "output": "Tas loco! Está demmas!"},
        {"input": "Que haces!?", "output": "Sos boludo! Que haces?"},
    ]

    # Prompt example
    # Here we tell the model what the input and output should look like
    example_prompt = ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    )

    # User input
    # Here we ask the user for input
    user_input = input("Enter a sentence: ")

    # build the few shot prompt
    # Here we build the few shot prompt
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # Here we build the messages
    _messages = [
        ("system", "You are an AI assistant that translates Español to Uruguayo."),
        few_shot_prompt,
        ("human", "{input}"),
    ]

    messages = ChatPromptTemplate.from_messages(messages=_messages).format(
        input=user_input
    )

    response = llm.invoke(messages)
    print(response.content)

    # using chains?
    messages = ChatPromptTemplate.from_messages(messages=_messages)
    chain = messages | llm
    response = chain.invoke({"input": user_input})
    print(response.content)


if __name__ == "__main__":
    main()
