#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: noob
"""

file_path = 'output.txt'

try:
    with open(file_path, 'r') as file:
        file_content = file.read()
except FileNotFoundError:
    print(f"File '{file_path}' not found.")
except IOError:
    print(f"Error reading from file '{file_path}'.")

data_list = file_content.split("########")

from langchain_core.documents.base import Document


documents = []
for content in data_list:
    document = Document(page_content=content)
    documents.append(document)
    
    

from langchain_community.embeddings import HuggingFaceEmbeddings
encode_kwargs = {'normalize_embeddings': [[False]]}
embedding_model_id = "asafaya/bert-large-arabic"

hf = HuggingFaceEmbeddings(
    model_name = embedding_model_id,
    encode_kwargs=encode_kwargs
)


from langchain_community.vectorstores import FAISS
print("creating database.....")
db = FAISS.from_documents(documents, hf)
print("database created successfully")

# retriever = db.as_retriever()
# docs = retriever.get_relevant_documents("ايه الاخبار يسطا")
# docs

print("saving database...")
db.save_local("faiss_db")
print("database saved successfully")
