from pipelines import pipeline
from utils import score_questions
from tableqa.agent import Agent
from tableqa.nlp import qa
import os
import textract
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
    
    def __text_process(self,text_dir):
        ext = str(text_dir).split(".")[-1]

        if ext=='txt':
            with open(text_dir,mode='r') as f:
                text = f.read()
        elif ext=='docx':
            text = textract.process(text_dir).decode('utf-8')
        else:
            raise Exception("Unsupported filetype")
        return text
    
    def get_results(self,text=None,text_dir=None,csv_dir=None,schema_dir=None):
        """
        

        Parameters
        ----------
        text : `str`
            Input text to be processed. Overridden if text_path is provided
        text_path : `str` or `pathlib.Path` object, path to file(.txt/.docx) containing textual information. Either text_path or text is required. 
        csv_path : `str` or `pathlib.Path` object, absolute path to folder containing all input files. Optional
        schema_path: `str` or `pathlib.Path` object, path to folder containing `json` schemas of input files. 
                    If not specified, auto-generated schema will be used.Optional


        Returns
        -------
        valmaps : `dict`
            Maps questions with generated answers.

        """
        
        valmaps={}
        
        if text_dir:
            text=self.__text_process(text_dir)
        
        if csv_dir:
            questions=self.get_questions(text,csv=True)
            agent=Agent(csv_dir,schema_dir)
            
            for q in questions:
                try:
                    res=agent.query_db(q)
                    valmaps[q]=res
                except:
                    pass
       
        else:
            questions=self.get_questions(text)
            for q in questions:
                valmaps[q]=qa(text,q)

        return valmaps
            
        
    
        
        
