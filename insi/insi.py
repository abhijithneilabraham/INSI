from pipelines import pipeline
from utils import score_questions
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
    
        
        
