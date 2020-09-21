from pipelines import pipeline
from utils import score_questions
from tableqa.agent import Agent
from tableqa.nlp import qa
import os
qg = pipeline("e2e-qg")


class insi:
    """
    Get insights from texts 
    """
    def get_scores(self,questions):
        """
        

        Parameters
        ----------
        questions : `list` or `tuple` of `str`
            Each string is a question.

        Returns
        -------
        `dict`
            Maps questions with scores

        """
        self.qmaps=score_questions(questions)
        return self.qmaps
    
    def get_questions(self,text,csv=False):
        """
        

        Parameters
        ----------
        text : `str`
            Input text to be processed.
        csv : Boolean, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        `list`
            `list` of strings containing questions .

        """
        questions=qg(text)
        if csv:
            qmaps=self.get_scores(questions)
            picked_questions=[k for k,v in qmaps.items() if v>0.9]
            return picked_questions
        return questions
    
    def get_results(self,text,csv_dir=None,schema_dir=None):
        """
        

        Parameters
        ----------
        text : `str`
            Input text to be processed.
        csv_path : `str` or `pathlib.Path` object, absolute path to folder containing all input files. Optional
        schema_path: `str` or `pathlib.Path` object, path to folder containing `json` schemas of input files. 
                    If not specified, auto-generated schema will be used.Optional


        Returns
        -------
        valmaps : `dict`
            Maps questions with generated answers.

        """
        
        valmaps={}
        if csv_dir:
            questions=self.get_questions(text,csv=True)
            agent=Agent(csv_dir,schema_dir)
            
            for q in questions:
                res=agent.query_db(q)
                valmaps[q]=res
       
        else:
            questions=self.get_questions(text)
            for q in questions:
                valmaps[q]=qa(text,q)

        return valmaps
            
        
    
        
        
