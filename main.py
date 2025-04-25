from langchain_ollama.llms import OllamaLLM
from prompt_builder import prompt
from vector import retriever

model = OllamaLLM(model="llama3") # Base Model from Ollama

chain = prompt | model # Pipeline using Langchain

exit_kw = ["/q", "/quit", "/exit"]

def chat():
    print("Personal Diabetes Assistent (LLM with RAG)")
    while True:
        question = input(">>> ")
        if question.lower() in exit_kw:
            break
        
        profiles = retriever.invoke(question) # Find relevant Vector DB entries
        result = chain.invoke({"profiles": profiles, "question": question}) # Run Pipeline
        print(result)
        print("\n")

if __name__ == "__main__":
    chat()