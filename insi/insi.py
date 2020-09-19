from pipelines import pipeline
from utils import score_questions
from tableqa.agent import Agent
import os
qg = pipeline("e2e-qg")


class insi:
    def get_scores(self,questions):
        self.qmaps=score_questions(questions)
        return self.qmaps
    
    def get_questions(self,text):
        questions=qg(text)
        qmaps=self.get_scores(questions)
        picked_questions=[k for k,v in qmaps.items() if v>0.9]
        return picked_questions
    
    def get_results(self,text,csv_path,schema=None):
        questions=self.get_questions(text)
        agent=Agent(csv_path,schema)
        valmaps={}
        for q in questions:
            res=agent.query_db(q)
            valmaps[q]=res
        return valmaps
        
    
        
        
