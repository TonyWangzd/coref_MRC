import json
import re

def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '
def pre_proc(text):
    text = re.sub(
        u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/|\t',
        space_extend, text)
    text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text

with open('./Bert4CoQA/coqa-dev-v1.0.json','r') as reader:
    context_data = json.load(reader)['data']
    for i in range(len(context_data)):
        context = context_data[i]['story']
        refine_context = pre_proc(context)
        with open('./CoQA_text.txt','a',encoding='utf-8') as f:
            f.write('/n'+refine_context)
            f.close