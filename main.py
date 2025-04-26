from langchain_ollama.llms import OllamaLLM
from prompt_builder import prompt
from vector import retriever

model = OllamaLLM(model="llama3") # Base Model from Ollama

chain = prompt | model # Pipeline using Langchain

exit_kw = ["/q", "/quit", "/exit"]

def chat():
    print("\033[1;36m==============================\033[0m")
    print("\033[1;32m Personal Diabetes Assistant\033[0m")
    print("\033[90m powered by LLM & RAG\033[0m")
    print("\033[1;36m==============================\033[0m")
    print("\033[90mType your question here (or type '/q', '/quit', '/exit' to quit):\n\033[0m")
    while True:
        question = input(">>> ")
        if question.lower() in exit_kw:
            break
        if question.lower()[0] == '/':
            print("\033[90mType your question here (or type '/q', '/quit', '/exit' to quit):\033[0m")
            continue
        
        profiles = retriever.invoke(question) # Find relevant Vector DB entries
        result = chain.invoke({"profiles": profiles, "question": question}) # Run Pipeline
        print(result)
        print("\n")

if __name__ == "__main__":
    chat()