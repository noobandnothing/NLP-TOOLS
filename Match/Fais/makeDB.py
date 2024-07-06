#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: noob
"""
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter


loader = TextLoader("output.txt")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=2550,chunk_overlap=8,separator="########")

docs = text_splitter.split_documents(documents)



from langchain_community.embeddings import HuggingFaceEmbeddings
encode_kwargs = {'normalize_embeddings': [[False]]}
embedding_model_id = "asafaya/bert-large-arabic"

hf = HuggingFaceEmbeddings(
    model_name = embedding_model_id,
    encode_kwargs=encode_kwargs
)


from langchain_community.vectorstores import FAISS
print("creating database.....")
db = FAISS.from_documents(docs, hf)
print("database created successfully")


print("saving database...")
db.save_local("faiss_db")
print("database saved successfully")
