#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 22:28:01 2020

@author: abhijithneilabraham
"""


from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer
from numpy import asarray,argmax
import os
import numpy as np
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
model=load_model("Question_Classifier.h5")


def score_questions(questions):
    scores={}
    for q in questions:
        print(q)
        emb=asarray(bert_model.encode([q]))
        scores[q]=np.amax(model.predict(emb))
    return scores


        
        
    


                
                

            

