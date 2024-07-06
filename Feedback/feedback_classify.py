#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:49:10 2024

@author: noob
"""

from camel_tools.sentiment import SentimentAnalyzer

sa = SentimentAnalyzer("CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment")


from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/rateFeedback', methods=['POST'])
def predict_nop():
    try:
        print(request.json)
        feedback = request.json.get('feedback', '')
        if not feedback:
            print("Invalid")
            raise Exception("Invalid Feedback")
        return jsonify({'feedback': feedback, 'result_pn':  sa.predict([feedback])[0] ,'status_code' :200})
    except Exception as e:
        print(e)
        return "",400
    
if __name__ == '__main__':
    app.run(host='192.168.0.105',port="6987")


