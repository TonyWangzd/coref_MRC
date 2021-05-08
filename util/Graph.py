import spacy
import string
nlp = spacy.load('en_core_web_sm')
import json
import re
import neuralcoref
neuralcoref.add_to_pipe(nlp)
import pandas as pd
#####graph lib
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

Coref_list = ['he', 'she', 'it', 'him', 'her', 'them', 'himself', 'herself', 'itself', 'its', 'I', 'me']

####change context to change the inpput
with open('./Bert4CoQA/coqa-dev-v1.0.json','r') as reader:
    context_data = json.load(reader)['data']
    context = context_data[1]['story']

####preprocess
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


def context_2node(context, sentence_number = None):
    #preprocess context
    refine_context = pre_proc(context)
    nlp_context = nlp(pre_proc(refine_context))
    annotated_text = process(nlp_context)
    #absorb node by sentence number
    graph_node = []
    if sentence_number:
        sentence_array = annotated_text['sentences'][sentence_number]
        sentence_text = nlp_context[sentence_array[0]:sentence_array[1]]
        print('sentence text is :', sentence_text)
        entity = entity_rec(sentence_text)
        for element in entity:
            graph_node.append(element)
        noun_chunk = [nc for nc in sentence_text.noun_chunks]
        for element in noun_chunk:
            graph_node.append(element)
    else:
        sentence_node = []
        for i in range(len(annotated_text['sentences'])):
            sentence_array = annotated_text['sentences'][i]
            sentence_text = nlp_context[sentence_array[0]:sentence_array[1]]
            print('sentence text is :', sentence_text)
            entity = entity_rec(sentence_text)
            for element in entity:
                sentence_node.append(element)
            noun_chunk = [nc for nc in sentence_text.noun_chunks]
            for element in noun_chunk:
                sentence_node.append(element)
            graph_node.append(sentence_node)
    return graph_node


def coref_window(windowsize, context):
    #preprocess context
    nlp_context = nlp(pre_proc(context))
    annotated_text = process(nlp_context)
    #absorb node by sentence number

    coref_list = []
    for i in range(len(annotated_text['sentences'])-windowsize):
        sentences = []
        for j in range(windowsize):
            sentence_array = annotated_text['sentences'][i+j]
            sentence_text = str(nlp_context[sentence_array[0]:sentence_array[1]])
            if j != windowsize-1:
                sentence_text = sentence_text.strip(string.punctuation)+','
            sentences.append(sentence_text)
        text = nlp(''.join(sentences))
        coref = text._.coref_clusters
        coref_list.append(coref)
    return coref_list


coref_list = coref_window(3, context)
node_list = context_2node(context)

def rebuild(one_list):
    #remove the repeat element
    return list(set(one_list))

def generating_coref_graph(coref_list,num):
    entity_pairs = []
    relation_pairs = []
    #coref_list = rebuild(coref_list)
    for i in range(1,len(coref_list)):
        # some times occurs Cotton and cotten it should be same
        if str(coref_list[0]).lower() == str(coref_list[i]).lower():
            relation_pairs.append('same')
        else:
            relation_pairs.append('coref')
        entity_pairs.append([str(coref_list[0]).lower()+'_'+str(num),str(coref_list[i]).lower()+'_'+str(num)])
    return entity_pairs, relation_pairs

def findGraph_Sentence(coref_list):
    final_entity_pairs = []
    final_relation_pairs = []
    cur_entity_pairs = []
    cur_relation_pairs = []
    for i in range(len(coref_list)):
        for j in range(len(coref_list[i])):
            cur_entity_pair, cur_relation_pair = generating_coref_graph(coref_list[i][j], i)
            for e in range(len(cur_entity_pair)):
                print('this is ',cur_entity_pair[e])
                if cur_entity_pair[e] not in final_entity_pairs and  [cur_entity_pair[e][1], cur_entity_pair[e][0]] not in final_entity_pairs:
                    final_entity_pairs.append(cur_entity_pair[e])
                    final_relation_pairs.append(cur_relation_pair[e])

    return final_entity_pairs, final_relation_pairs
    
# to filter the pairs, remove the same element and 
def add_pairs(final_entity_pairs, final_relation_pairs):
    solve_number = 0
    for i in range(len(final_entity_pairs)):
        current_pair = final_entity_pairs[i]
        cur_left = current_pair[0].split('_',1)[0]
        cur_right = current_pair[1].split('_',1)[0]
        cur_number = current_pair[0].split('_',1)[1]
        if cur_left in Coref_list and cur_right in Coref_list and int(cur_number) >= solve_number:
            count = i
            stop = False
            while(not stop and count >=0):
                count -= 1
                check_pair = final_entity_pairs[count]
                check_left = check_pair[0].split('_',1)[0]
                check_right= check_pair[1].split('_',1)[0]
                check_number = check_pair[0].split('_',1)[1]
                if cur_left == check_right and check_left not in Coref_list:
                    stop = True
                    # add the cotton coref her in new line
                    new_left = check_left+'_'+cur_number
                    new_right= check_right+'_'+cur_number
                    final_entity_pairs.insert(i, [new_left, new_right])    
                    final_relation_pairs.insert(i, 'coref')
                    # add cotton same cotton in new line
                    new_left2 = check_left+'_'+check_number
                    new_right2 = check_left+'_'+cur_number
                    final_entity_pairs.insert(count, [new_left2, new_right2])
                    final_relation_pairs.insert(count, 'same')
                    solve_number = int(cur_number) + 1

    return final_entity_pairs, final_relation_pairs

def refine_the_same(entity_pairs, relation_pairs):
    i = 0
    while i < len(entity_pairs):
        if entity_pairs[i][0] == entity_pairs[i][1]:
            del entity_pairs[i]
            del relation_pairs[i]
            i -= 1
        i += 1

###generate graph
def create_graph(context, num):
    coref_list = coref_window(3, context)
    final_entity_pairs, final_relation_pairs = findGraph_Sentence(coref_list)
    add_entity_pairs, add_relation_pairs = add_pairs(final_entity_pairs, final_relation_pairs)
    refine_the_same(add_entity_pairs, add_relation_pairs)

    # create by graph
    # extract subject
    source = [i[0] for i in add_entity_pairs]
    # extract object
    target = [i[1] for i in add_entity_pairs]
    kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':add_relation_pairs})
    G=nx.from_pandas_edgelist(kg_df,source="source", target="target")
    plt.figure(figsize=(24,24))
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos, font_size = 10, linewidths=20)
    plt.show()
    plt.savefig('./result_graph/test_'+str(num)+'.jpg')
    kg_df.to_csv('./result_csv/test_'+str(num)+'.csv')

if __name__ == "__main__":
    with open('./Bert4CoQA/coqa-dev-v1.0.json','r') as reader:
        context_data = json.load(reader)['data']
    for i in range(len(context_data)):
        context = context_data[i]['story']
        create_graph(context, i)