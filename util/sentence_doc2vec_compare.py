import gensim.models as gm
import nltk
import numpy as np
from scipy.linalg import norm


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

if __name__ == "__main__":
    sent1 = u'there lived a little white kitten named Cotton.'
    sent2 = u"Cotton lived high up in a nice warm place above the barn where all of the farmer's horses slept."
    print(vector_similarity(sent1, sent2))