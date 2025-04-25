from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

db_location = "./vector_db"
collection_name = "patient_profiles"

embeddings = OllamaEmbeddings(model="mxbai-embed-large") # Model for embeddings

df = pd.read_csv("diabetes_dataset.csv")[:1000] # Optional: Change size of Vector DB

add_documents = not os.path.exists(db_location) # Check for existing Vector DB

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        profile = (
            f"Patient #{row['id']} here are the following informations: "
            f"{row['Age']} year old {row['Sex'].lower()} from {row['Ethnicity']} "
            f"with BMI {row['BMI']}, waist {row['Waist_Circumference']} cm, "
            f"fasting glucose {row['Fasting_Blood_Glucose']} mg/dL, "
            f"HbA1c {row['HbA1c']}%, blood pressure {row['Blood_Pressure_Systolic']}/{row['Blood_Pressure_Diastolic']} mmHg, "
            f"total cholesterol {row['Cholesterol_Total']} mg/dL, HDL {row['Cholesterol_HDL']} mg/dL, "
            f"LDL {row['Cholesterol_LDL']} mg/dL, GGT {row['GGT']}, serum urate {row['Serum_Urate']}. "
            f"Physical activity: {row['Physical_Activity_Level']}, alcohol consumption: {row['Alcohol_Consumption']}, "
            f"smoking status: {row['Smoking_Status']}. "
            f"Family history of diabetes: {row['Family_History_of_Diabetes']}, "
            f"previous gestational diabetes: {row['Previous_Gestational_Diabetes']}."
        )

        document = Document(
            page_content=profile,
            metadata={}, # Optional add metadata
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
