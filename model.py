import torch
from torch import nn
import torch.nn.functional as F


class TextRNN(nn.Module):
    def __init__(self,conf):
        super(TextRNN,self).__init__()
        self.embedding=nn.Embedding(num_embeddings=conf.n_vocab,
                                    embedding_dim=conf.embed_size,
                                    padding_idx=conf.n_vocab-1)
        self.lstm=nn.LSTM(input_size=conf.embed_size,
                          hidden_size=conf.hidden_size,
                          num_layers=conf.num_layers,
                          batch_first=True,
                          dropout=conf.dropout,
                          bidirectional=True)
        # self.maxpool=nn.MaxPool1d(conf.pad_size)#默认stride与kernel_size相同
        self.avgpool=nn.AdaptiveMaxPool1d(output_size=1)
        self.fc=nn.Linear(in_features=2*conf.hidden_size+conf.embed_size,out_features=64)
        self.fc1=nn.Linear(in_features=64,out_features=conf.num_classes)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,x):
        embed=self.embedding(x)# [B,seqlen]--->[B,seqlen,embed_size]
        out,_=self.lstm(embed,None)
        out=torch.cat([embed,out],dim=2)
        out=F.relu(out)
        out=out.permute(0,2,1)
        # out=self.maxpool(out)
        out=self.avgpool(out)
        out=out.reshape(out.size(0),-1)
        out1=self.fc(out)
        out=self.fc1(out1)
        # out=self.softmax(out)
        return out,out1 #两路输出，分类logits和进入分类器前的特征

class GCNN(nn.Module):
    def __init__(self,conf):
        super(GCNN,self).__init__()
        self.embed=nn.Embedding(conf.n_vocab,conf.embed_size)
        nn.init.xavier_uniform(self.embed.weight)
        self.conva_1=nn.Conv1d(conf.embed_size,64,15,stride=7)
        self.convb_1=nn.Conv1d(conf.embed_size,64,15,stride=7)
        self.conva_2=nn.Conv1d(64,64,15,stride=7)
        self.convb_2=nn.Conv1d(64,64,15,stride=7)
        self.out_linear1=nn.Linear(19,64)
        self.out_linear2=nn.Linear(64,conf.num_classes)
    def forward(self,x):
        #编码[B,max_seq_len]->[B,max_seq_len,embedding_dim]
        x=self.embed(x)
        #第一层卷积门
        x=x.transpose(1,2) #[B,embedding_dim,max_seq_len]
        A=self.conva_1(x)
        B=self.convb_1(x)
        H=A*torch.sigmoid(B)#[B,64,max_sql_len]
        A=self.conva_2(H)
        B=self.convb_2(H)
        H=A*torch.sigmoid(B)#[B,64,max_sql_len]
        pool_out=torch.mean(H,dim=1)#平均池化，[B,64]
        linear1_output=self.out_linear1(pool_out)
        logits=self.out_linear2(linear1_output)#[B,num_classes]
        return logits,linear1_output

class FastText(nn.Module):
    def __init__(self,conf):
        super(FastText,self).__init__()
        self.embed = nn.Embedding(conf.n_vocab,conf.embed_size)  # embedding初始化，需要两个参数，词典大小、词向量维度大小
        self.embed.weight.requires_grad = True  # 需要计算梯度，即embedding层需要被训练

        self.fc1=nn.Linear(conf.embed_size, conf.hidden_size)  # 这里的意思是先经过一个线性转换层
        self.bn=nn.BatchNorm1d(conf.hidden_size)  # 再进入一个BatchNorm1d
        self.relu=nn.ReLU(inplace=True)  # 再经过Relu激活函数
        self.fc2=nn.Linear(conf.hidden_size, conf.num_classes)  # 最后再经过一个线性变换

    def forward(self, x):
        x = self.embed(x)  # 先将词id转换为对应的词向量
        x=torch.mean(x,dim=1)
        x1=self.fc1(x)
        x2=self.bn(x1)
        x2=self.relu(x2)
        out=self.fc2(x2)
        return out,x1

class Confus(nn.Module):
    def __init__(self,conf):
        super(Confus,self).__init__()
        self.embedding = nn.Embedding(num_embeddings=conf.n_vocab,embedding_dim=conf.embed_size,padding_idx=conf.n_vocab - 1)
        #定义网络1
        self.lstm = nn.LSTM(input_size=conf.embed_size,hidden_size=conf.hidden_size,
                            num_layers=conf.num_layers,batch_first=True,
                            dropout=conf.dropout,bidirectional=True)
        self.avgpool = nn.AdaptiveMaxPool1d(output_size=1)
        self.fc = nn.Linear(in_features=2 * conf.hidden_size + conf.embed_size, out_features=64)
        #定义网络2
        self.conva_1 = nn.Conv1d(conf.embed_size, 64, 15, stride=7)
        self.convb_1 = nn.Conv1d(conf.embed_size, 64, 15, stride=7)
        self.conva_2 = nn.Conv1d(64, 64, 15, stride=7)
        self.convb_2 = nn.Conv1d(64, 64, 15, stride=7)
        self.out_linear1 = nn.Linear(19, 64)
        #定义网络3
        self.fc1 = nn.Linear(conf.embed_size, conf.hidden_size)  # 这里的意思是先经过一个线性转换层
        #定义分类层
        self.fc3=nn.Linear(64*3,conf.num_classes)

    def forward(self,x):
        embed = self.embedding(x)  # [B,seqlen]--->[B,seqlen,embed_size]
        #分支1
        out, _ = self.lstm(embed, None)
        out = torch.cat([embed, out], dim=2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        # out=self.maxpool(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out1 = self.fc(out)
        #分支2
        x = embed.transpose(1, 2)  # [B,embedding_dim,max_seq_len]
        A = self.conva_1(x)
        B = self.convb_1(x)
        H = A * torch.sigmoid(B)  # [B,64,max_sql_len]
        A = self.conva_2(H)
        B = self.convb_2(H)
        H = A * torch.sigmoid(B)  # [B,64,max_sql_len]
        pool_out = torch.mean(H, dim=1)  # 平均池化，[B,64]
        linear1_output = self.out_linear1(pool_out)
        #分支3
        out2 = torch.mean(embed, dim=1)
        out2 = self.fc1(out2)
        confus=torch.cat([out1,linear1_output,out2],dim=1)
        y=self.fc3(confus)
        return (y,0)#第2个元素无意义，只是为了和训练脚本对应

if __name__ == '__main__':
    from config import Config
    conf=Config()
    conf.pad_size=640
    net=TextRNN(conf)
    input_tensor=torch.tensor([i for i in range(640)]).reshape([1,640])
    out=net.forward(input_tensor)
    print(out.size())








