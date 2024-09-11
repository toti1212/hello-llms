import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq

_ = load_dotenv(find_dotenv())


GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def main():
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=GROQ_API_KEY,
    )

    messages = [
        (
            "system",
            "You are a expert in python programming and you are tasked to help me with my code.",
        ),
        (
            "human",
            "What is the best way to learn python architecture as a professional developer from a big tech company?",
        ),
    ]
    response = llm.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    main()
