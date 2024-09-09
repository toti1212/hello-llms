import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAI, ChatOpenAI

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")


def chatgpt_api():
    llm = OpenAI(
        model="gpt-3.5-turbo-instruct",
        organization=OPENAI_ORGANIZATION,
        api_key=OPENAI_API_KEY,
        temperature=0.1,  # how creative the LLM is
    )
    print("Calling the LLM...")
    try:
        response = llm.invoke("Say hi to Rodrigo")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

    try:
        print("Calling the LLM in streaming mode...")
        for chunk in llm.stream("Say hi to Rodrigo"):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"Error: {e}")

def chatgpt_chat():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        organization=OPENAI_ORGANIZATION,
        api_key=OPENAI_API_KEY,
    )
    
    messages = [
        {"role": "system", "content": "You are an expert in python programming and you are tasked to help me with my code."},
        {"role": "human", "content": "How is the best way to learn python architecture as a professional developer from a big tech company?"},
    ]
    try:
        # response = llm.invoke(messages)
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
    except Exception as e:
        print(f"Error: {e}")

def main():
    # chatgpt_api()
    chatgpt_chat()


if __name__ == "__main__":
    main()
