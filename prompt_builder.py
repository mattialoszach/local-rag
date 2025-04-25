from langchain_core.prompts import ChatPromptTemplate

template = """
You are an expert medical assistant specialized in analyzing patient profiles.

Here are some relevant patient profiles:
{profiles}

Based on this context, answer the following question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)