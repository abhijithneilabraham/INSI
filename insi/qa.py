


import os
from transformers import TFBertForQuestionAnswering, BertTokenizer
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from rake_nltk import Rake
import json


qa_model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
qa_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad',padding=True)


import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
lem=lemmatizer.lemmatize




def extract_keywords_from_doc(doc, phrases=True, return_scores=False):
    if phrases:
        r = Rake()
        if isinstance(doc, (list, tuple)):
            r.extract_keywords_from_sentences(doc)
        else:
            r.extract_keywords_from_text(doc)
        if return_scores:
            return [(b, a) for a, b in r.get_ranked_phrases_with_scores()]
        else:
            return r.get_ranked_phrases()
    else:
        if not isinstance(doc, (list, tuple)):
            doc = [doc]
        ret = []
        for x in doc:
            for t in nltk.word_tokenize(x):
                if t.lower() not in stop_words:
                    ret.append(t)
        return ret


def extract_keywords_from_query(query, phrases=True):
    if not phrases:
        tokens = nltk.pos_tag(nltk.word_tokenize(query))
        return [t[0] for t in tokens if  t[0].lower() not in stop_words and t[1] != '.']
    kws = extract_keywords_from_doc(query, phrases=True)
    tags = dict(nltk.pos_tag(nltk.word_tokenize(query)))
    filtered_kws = []
    for kw in kws:
        kw_tokens = nltk.word_tokenize(kw)
        for t in kw_tokens:
            if t in tags and tags[t][0] in ('N', 'C', 'R', 'S'):
                filtered_kws.append(kw)
                break
    return filtered_kws


def qa(docs, query, return_score=False, return_all=False, return_source=False, sort=False):
    if isinstance(docs, (list, tuple)):
        answers_and_scores = [qa(doc, query, return_score=True) for doc in docs]
        if sort:
            sort_ids = list(range(len(docs)))
            sort_ids.sort(key=lambda i: -answers_and_scores[i][1])
            answers_and_scores = [answers_and_scores[i] for i in sort_ids]
        if return_source and sort:
            docs = [docs[i] for i in sort_ids]
        if not return_score:
            answers = [a[0] for a in answers_and_scores]
        else:
            answers = answers_and_scores
        if return_source:
            if return_score:
                answers = [answers[i] + (docs[i],) for i in range(len(docs))]
            else:
                answers = [(answers[i], docs[i]) for i in range(len(docs))]
        return answers if return_all else answers[0]

    doc = docs
    input_ids = qa_tokenizer.encode(query, doc)

    if len(input_ids) > 512:
        sentences = nltk.sent_tokenize(doc)
        if len(sentences) == 1:
            if return_score:
                return '', -1000
            else:
                return ''
        else:
            return qa(sentences, query, return_score=return_score)
    sep_index = input_ids.index(qa_tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    start_scores, end_scores = qa_model(tf.constant([input_ids]),
                                 token_type_ids=tf.constant([segment_ids]))
    
    tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
    num_input_tokens = sep_index + 1
    answer_start = tf.argmax(start_scores[0][num_input_tokens:]) + num_input_tokens
    answer_end = tf.argmax(end_scores[0][num_input_tokens:]) + num_input_tokens

    answer = ' '.join(tokens[int(answer_start.numpy()): int(answer_end.numpy() + 1)])
    answer = answer.replace(' #', '').replace('#', '').replace('[CLS]', '').replace('[SEP]', '')
    if not return_score:
        return answer
    input_kws = set(extract_keywords_from_query(query.lower(), phrases=False))
    answer = answer.replace(' #', '').replace('#', '').replace('[CLS]', '').replace('[SEP]', '')
    answer_kws = set(extract_keywords_from_query(answer.lower(), phrases=False))
    num_input_kws = len(input_kws)
    input_kws.update(answer_kws)
    if len(input_kws) == num_input_kws:
        score = 0
    else:
        score = float((start_scores[0][answer_start] + end_scores[0][answer_end]))
    return answer, score

