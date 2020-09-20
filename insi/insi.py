from pipelines import pipeline
from utils import score_questions
from tableqa.agent import Agent
from tableqa.nlp import qa
import os
qg = pipeline("e2e-qg")


class insi:
    def get_scores(self,questions):
        self.qmaps=score_questions(questions)
        return self.qmaps
    
    def get_questions(self,text,csv=False):
        questions=qg(text)
        if csv:
            qmaps=self.get_scores(questions)
            picked_questions=[k for k,v in qmaps.items() if v>0.9]
            return picked_questions
        return questions
    
    def get_results(self,text,csv_path=None,schema=None):
       
        
        valmaps={}
        if csv_path:
            questions=self.get_questions(text,csv=True)
            agent=Agent(csv_path,schema)
            
            for q in questions:
                res=agent.query_db(q)
                valmaps[q]=res
       
        else:
            questions=self.get_questions(text)
            for q in questions:
                valmaps[q]=qa(text,q)

        return valmaps
            
        
    
        
        
