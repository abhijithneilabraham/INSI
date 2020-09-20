# INSI
Text analysis tool to provide insights. 

Automatically generates questions and answers from an input text  to provide insights.

Built on top of :   
[Question Generator](https://github.com/patil-suraj/question_generation) :Generates questions from input text.   
[tableQA](https://github.com/abhijithneilabraham/tableQA) :Applies the natural language queries to csv files.


### Configuration

```git clone https://github.com/abhijithneilabraham/INSI/```

```cd insi```

### Quickstart
#### Text insights

```
from insi import insi
nlp=insi()
text="A large number of cancer patients died.Many deaths were caused by stomach cancer."
nlp.get_results(text)
#{'How many cancer patients died?': 'a large number', 'How many deaths were caused by stomach cancer?': 'many'}
```
#### Improved insights with csvs

Using csv file(s) related to the text could help build more insights.The sample input csvs and schemas can be found in [sample](insi/sample) and [schema](insi/schema) respectively. Using a schema is optional.   


```
from insi import insi
nlp=insi()
text="A large number of cancer patients died.Many deaths were caused by stomach cancer."
nlp.get_results(text,csv_path,schema_path)

#{'How many cancer patients died?': [(2431,)], 'How many deaths were caused by stomach cancer?': [(179,)]}
```




