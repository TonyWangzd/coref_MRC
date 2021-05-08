import spacy
import string
nlp = spacy.load('en_core_web_sm')
import json
import re
############
import gensim.models as gm
import nltk
import numpy as np
from scipy.linalg import norm
import random


model_file = './Bert4CoQA/model/CoQA_en_doc2vecModel.bin'
#model_file = './model/CoQA_en_doc2vecModel.bin'
model = gm.Doc2Vec.load(model_file)

def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = nltk.word_tokenize(s)
        v = np.zeros(256)
        v = model.infer_vector(words)
        v /= len(words)
        return v
    
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))
##################

with open('./Bert4CoQA/coqa-dev-v1.0.json','r') as reader:
    context_data = json.load(reader)['data']
    context = context_data[0]['story']

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '
def pre_proc(text):
    text = re.sub(
        u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t',
        space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

def _str(s):
    """ Convert PTB tokens to normal tokens """
    if (s.lower() == '-lrb-'):
        s = '('
    elif (s.lower() == '-rrb-'):
        s = ')'
    elif (s.lower() == '-lsb-'):
        s = '['
    elif (s.lower() == '-rsb-'):
        s = ']'
    elif (s.lower() == '-lcb-'):
        s = '{'
    elif (s.lower() == '-rcb-'):
        s = '}'
    return s

def process(parsed_text):
    output = {'word': [], 'offsets': [], 'sentences': []}

    for token in parsed_text:
        output['word'].append(_str(token.text))
        output['offsets'].append((token.idx, token.idx + len(token.text)))

    word_idx = 0
    for sent in parsed_text.sents:
        output['sentences'].append((word_idx, word_idx + len(sent)))
        word_idx += len(sent)

    assert word_idx == len(output['word'])
    return output

def entity_rec(text):
    ners = [ent.text for ent in text.ents]
    return ners

def get_sentence(context):
    sentences = []
     #preprocess context
    refine_context = pre_proc(context)
    nlp_context = nlp(pre_proc(refine_context))
    annotated_text = process(nlp_context)
    for i in range(len(annotated_text['sentences'])):
        sentence_array = annotated_text['sentences'][i]
        sentence_text = nlp_context[sentence_array[0]:sentence_array[1]]
        sentences.append(str(sentence_text))            
    return sentences

def mk_binding_file(context_data):
    sentences = get_sentence(context_data['story'])
    questions_text = context_data['questions']
    evidence_re = [None*len(sentences) for i in range(len(questions_text))]
    for i in range(len(questions_text)):
        cur_question = questions_text[i]['input_text']
        for j in range(len(sentences)):
            evidence_re[i][j] = vector_similarity(cur_question, sentences[j])
    return evidence_re

def generate_data(data, index): 
    context = data['story']
    sentences = get_sentence(context)
    questions_text = data['questions']
    ground_true = data['additional_answers']
    count = 0

    for i in range(len(questions_text)):
        answer_list = []
        lable_list = []
        current_question = questions_text[i]['input_text']
        current_True_answer = ground_true[i]['span_text'] 
        for j in sentences:
            if current_True_answer in j:
                current_True_answer = j
                answer_list.append(current_True_answer)
                lable_list.append(1)
        if current_True_answer in sentences:
            rest = sentences.copy().remove(current_True_answer)
        else:
            rest = sentences

        current_False_answer = sentences[random.randint(0, len(rest))]
        answer_list.append(current_False_answer)
        lable_list.append(0)
        
        pair_df = pd.DataFrame({'question': current_question, 'evidence':answer_list, 'lable':lable_list})
        pair_df.to_csv('./test_evidence.csv')


mk_binding_file(context_data[0])