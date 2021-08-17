import string
import json
import re
import pandas as pd
#####graph lib
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
#####AllenNLP
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
####Spacy
import spacy
nlp = spacy.load('en_core_web_sm')


#########preprocess and parse########
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
#########Graph construction########

def judge_sentence_number(word_token, sentence_list):
    #judge which sentence the word_token belongs
    for i in range(len(sentence_list)):
        if word_token >= sentence_list[i][0] and word_token <= sentence_list[i][1]:
            return i

def output_with_tag(document):
    #the function defined as:
    #input: parsed context's json
    #output: each cluster with sentence number on it
    #preprocess context
    nlp_context = nlp(pre_proc(document))
    coref_parsed_json = predictor.predict(str(nlp_context))
    annotated_text = process(nlp_context)
    
    document_clusters = coref_parsed_json['clusters']
    document_tokens = coref_parsed_json['document']
    sentences_list = annotated_text['sentences']

    #save the result in entity_pair and relation_pair
    entity_pairs = []
    relation_pairs = []
    
    for coref_cluster in document_clusters:
        main_word_token = coref_cluster[0]
        main_word_instance = ' '.join(document_tokens[main_word_token[0]:main_word_token[1]+1])
        #judge the sentence number
        main_word_sentence_index = judge_sentence_number(main_word_token[0], sentences_list)
        
        for i in range(1, len(coref_cluster)):
            coref_word_token = coref_cluster[i]
            coref_word_instance = ' '.join(document_tokens[coref_word_token[0]:coref_word_token[1]+1])
            coref_word_sentence_index = judge_sentence_number(coref_word_token[0], sentences_list)
            
            #add into relation pair
            if str(main_word_instance).lower() == str(coref_word_instance).lower():
                relation_pairs.append('same')
            else:
                relation_pairs.append('coref')
            entity_pairs.append([str(main_word_instance).lower()+'_'+str(main_word_sentence_index),str(coref_word_instance).lower()+'_'+str(coref_word_sentence_index)])

    return relation_pairs, entity_pairs

def create_graph(context, num):
    #create the entity and relation for coreference graph
    relation_pairs, entity_pairs = output_with_tag(context)

    # create by graph
    # extract subject
    source = [i[0] for i in entity_pairs]
    # extract object
    target = [i[1] for i in entity_pairs]
    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relation_pairs})
    G=nx.from_pandas_edgelist(kg_df,source="source", target="target")
    plt.figure(figsize=(24,24))
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos, font_size = 10, linewidths=20)
    plt.show()
    plt.savefig('./result_graph/test_'+str(num)+'.jpg')
    kg_df.to_csv('./result_csv/test_'+str(num)+'.csv')
            
if __name__ == "__main__":
    with open('/Users/wangzd/Desktop/code/CoQA/coqa-dev-v1.0.json','r') as reader:
        context_data = json.load(reader)['data']
    for i in range(len(context_data)):
        context = context_data[i]['story']
        create_graph(context, i)