###########data processing##############
import spacy
import string
nlp = spacy.load('en_core_web_sm')
import json
import re
import neuralcoref
neuralcoref.add_to_pipe(nlp)

with open('/ssd2/wangzd/code/Bert4CoQA/coqa-dev-v1.0.json','r') as reader:
    context_data = json.load(reader)['data']

data_PATH = '/ssd2/wangzd/code/outpt/assessment/dev/test_evidence_with_lable.csv'
test_PATH = '/ssd2/wangzd/code/outpt/assessment/dev/test_evidence_with_lable_dev.csv'

###########machine Learning#############
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torch.nn.functional as F

import random
import pandas as pd
from tqdm import tqdm

import gensim.models as gm
import nltk
import numpy as np
from scipy.linalg import norm

CHAR_SIZE=2000
embedding_size=2000
maxlen=30
EPOCH=50
BATCH_SIZE=20
LR=0.001

#model_file = './Bert4CoQA/model/CoQA_en_doc2vecModel.bin'
model_file = '/ssd2/wangzd/code/Bert4CoQA/backpack/comparation_query_context/model/CoQA_en_doc2vecModel.bin'
model = gm.Doc2Vec.load(model_file)


def sentence_vector(s):
    words = nltk.word_tokenize(s)
    v = np.zeros(256)
    v = model.infer_vector(words)
    v /= len(words)
    return v

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


def load_data(path):
    # use this when the function packed
    context_mt = []
    for i in tqdm(range(len(context_data))):
        context = context_data[i]['story']
        sentences = get_sentence(context)
        context_mt.append([(sentence_vector(sent)) for sent in sentences])
    context_mt = np.array(padding(context_mt))

    data = pd.read_csv(path)
    query_mt = []
    evidence_mt = []
    cur_context_mt = []

    for i in tqdm(range(len(data))):
        index = int(data['index'][i])
        # only use the data when label is less than 30
        if data['lable'][i] < maxlen:
            query_mt.append(sentence_vector(data['question'][i]).reshape(1, CHAR_SIZE))
            evidence_mt.append(data['lable'][i])     
            cur_context_mt.append(context_mt[index])
    return np.array(cur_context_mt), np.array(evidence_mt), np.array(query_mt)

def padding(text,maxlen):
    pad_text=[]
    for line in text:
        #建立一个形状符合输出的列表
        pad_sentence=np.zeros((maxlen, CHAR_SIZE))
        cnt=0
        for index in line:
            pad_sentence[cnt]+=index
            cnt+=1
            if cnt== maxlen:
                break
        pad_text.append(pad_sentence)
    return pad_text

class DSSM(torch.nn.Module):
    def __init__(self):
        super(DSSM,self).__init__()
        self.embedding=nn.Embedding(CHAR_SIZE,embedding_size)
        self.linear1=nn.Linear(embedding_size,256)
        self.linear2=nn.Linear(256,128)
        self.linear3=nn.Linear(128,64)
        self.dropout=nn.Dropout(p=0.2)
        
    def forward(self,a,b):
        #将各索引id的embedding向量相加
        a=self.embedding(a).sum(1)
        b=self.embedding(b).sum(1)
        
        a=torch.tanh(self.linear1(a))
        a=self.dropout(a)
        a=torch.tanh(self.linear2(a))
        a=self.dropout(a)
        a=torch.tanh(self.linear3(a))
        a=self.dropout(a)
        
        b=torch.tanh(self.linear1(b))
        b=self.dropout(b)
        b=torch.tanh(self.linear2(b))
        b=self.dropout(b)
        b=torch.tanh(self.linear3(b))
        b=self.dropout(b)
        
        cosine=torch.cosine_similarity(a,b,dim= 1,eps=1e-8)
        return cosine
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

class CoQADataset(Dataset):
    def __init__(self,filepath):
        self.path=filepath
        self.context_mt,self.query_mt,self.evidence_mt = load_data(filepath)
    def __getitem__(self, idx):
        return self.context_mt[idx],self.query_mt[idx], self.evidence_mt[idx]
    def __len__(self):
        return len(self.query_mt)

##########main process##########
if __name__ == '__main__':
    train_data=CoQADataset(data_PATH)
    test_data=CoQADataset(test_PATH)

    #1、创建数据集并创立数据载入器
    #注意修改测试集
    train_loader=DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
    test_loader=DataLoader(dataset=test_data,batch_size=BATCH_SIZE,shuffle=True)

    #2、有gpu用gpu，否则cpu
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    dssm=DSSM().to(device)
    dssm._initialize_weights()

    #3、定义优化方式和损失函数
    optimizer=torch.optim.Adam(dssm.parameters(),lr=LR)
    loss_func=nn.CrossEntropyLoss()

############Trainning##########
    for epoch in tqdm(range(EPOCH)):
        for step,(context,label,query) in tqdm(enumerate(train_loader)):
            #1、把索引转化为tensor变量，载入设备，注意转化成long tensor
            a=Variable(context.to(device).long())
            b=Variable(query.to(device).long())
            l=Variable(label.to(device).long())
            #2、计算余弦相似度
            # choose every element in to calculate

            re = torch.stack([dssm(a[i],b[i]) for i in range(len(a))],0).to(device)
            #3、预测结果传给loss
            out = F.softmax(re, dim = 1)
            loss = loss_func(out,l)
            print('[Step]:',step+1,'训练loss:',loss.item())
            
            #4、固定格式
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (step+1) % 100 == 0:
                total=0
                correct=0
                for (test_a,test_l,test_b) in tqdm(test_loader):
                    tst_a=Variable(test_a.to(device).long())
                    tst_b=Variable(test_b.to(device).long())
                    tst_l=Variable(test_l.to(device).long())
                    tst_re = torch.stack([dssm(tst_a[i],tst_b[i]) for i in range(len(tst_a))],0).to(device)
                    out=torch.max(F.softmax(tst_re, dim=1),1)[1]
                    if out.size()==tst_l.size():
                        total+=tst_l.size(0)
                        correct+=(out==tst_l).sum().item()

                print('[Epoch]:',epoch+1,'训练loss:',loss.item())
                print('[Epoch]:',epoch+1,'测试集准确率: ',(correct*1.0/total))


