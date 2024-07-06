#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: noob
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


print("loading embedding model and index ....")

encode_kwargs = {'normalize_embeddings': False}
embedding_model_id = "asafaya/bert-large-arabic"

embeddings = HuggingFaceEmbeddings(
    model_name = embedding_model_id,
    encode_kwargs=encode_kwargs
)


def getRelavent(question):
    db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(question)
    # for doc  in docs:
    #     print(doc)
    return docs[0].page_content

def generate_prompt(question, context):
    return f"""
        انت مساعد مفيد متخصص في الاجابة عن أسئلة عن السيرة النبوية لسيدنا محمد صلي الله عليه وسلم. 
        مهمتك هي الاجابة عن السؤال التالي: '{question}' 
        في ضوء القطعة التالية: '{context}'
        
        الاجابة هي: 
    """
