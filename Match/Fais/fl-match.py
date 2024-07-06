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
    

from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/getprompt', methods=['POST'])
def retrieve_documents_prompt():
    try:
        print(request.json)
        question = request.json.get('question', '')
        if not question:
            print("Invalid")
            raise Exception("Invalid Question")
        
        content = getRelavent(question)
        prompt = generate_prompt(question,content)
        
        return jsonify({'question': question, 'prompt': prompt ,'status_code' :200})
    except Exception as e:
        print(e)
        return "",400

@app.route('/getmatch', methods=['POST'])
def retrieve_documents():
    try:
        print(request.json)
        question = request.json.get('question', '')
        if not question:
            print("Invalid")
            raise Exception("Invalid Question")
        
        content = getRelavent(question)
        content = content.split("الإجابة:")[1]
        
        return jsonify({'question': question, 'response': content ,'status_code' :200})
    except Exception as e:
        print(e)
        return "",400
    
if __name__ == '__main__':
    app.run(host='192.168.1.101',port="5987")
