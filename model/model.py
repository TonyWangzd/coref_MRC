import torch
import torch.nn as nn
import args

class DSSM(nn.Module):
    def __init__(self):
        super(DSSM,self).__init__()
        self.embedding=nn.Embedding(args.CHAR_SIZE,args.EMBEDDING_SIZE)
        self.linear1=nn.Linear(args.EMBEDDING_SIZE,256)
        self.linear2=nn.Linear(256,128)
        self.linear3=nn.Linear(128,64)
        self.dropout=nn.Dropout(p=0.2)
        
    def forward(self,a,b):
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
        
        cosine=torch.cosine_similarity(a,b,dim=1,eps=1e-8)
        return cosine

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)