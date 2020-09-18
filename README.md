# INSI
Text analysis tool to provide insights. 

Combines the information from a text and csv to provides insights.

Built on top of :
[Question Generator](https://github.com/patil-suraj/question_generation) #Generates questions from input text.
[tableQA](https://github.com/abhijithneilabraham/tableQA) #Applies the natural language queries to csv files.


### Configuration

```git clone https://github.com/abhijithneilabraham/INSI/```

```cd insi```

```
from insi import insi
nlp=insi()
text="A large number of cancer patients died.Many deaths were caused by stomach cancer."
nlp.get_results(text,csv_path)
```




