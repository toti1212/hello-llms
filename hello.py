import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")

def main():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        organization=OPENAI_ORGANIZATION,
        api_key=OPENAI_API_KEY,
    )
    print("Calling the LLM...")
    try:
        response = llm.invoke("Say hi to Rodrigo")
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    main()
