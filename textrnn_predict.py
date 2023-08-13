
import torch
from torch.utils.data import DataLoader
from model import TextRNN,GCNN,FastText
from  mydataset import MyDataSet
from config import Config
import numpy as np


cfg=Config()

train_dataset=MyDataSet(voc_dict_path='data/dict',data_path='data/train.txt',stop_word_path='data/hit_stopwords.txt')
cfg.pad_size=train_dataset.max_sql_len
train_loader=DataLoader(train_dataset,batch_size=1)

val_dataset=MyDataSet(voc_dict_path='data/dict',data_path='data/test.txt',stop_word_path='data/hit_stopwords.txt')
val_loader=DataLoader(val_dataset,batch_size=1)

model=TextRNN(cfg)
model.load_state_dict(torch.load('checkpoints/TextRNN.pth'))
model.to(cfg.device)
model.eval()
train_file='textrnn_features_seq1000/train_feature.txt'
test_file='textrnn_features_seq1000/test_feature.txt'
with torch.no_grad():
    train_data=[]
    for batch in train_loader:
        label, data = batch
        label, data = label.to(cfg.device), data.to(cfg.device)
        logits = model.forward(data)
        feature=logits[1].cpu().numpy().reshape(-1)
        line=np.insert(feature,0,label.item()).tolist()
        train_data.append(line)
    train_data=np.array(train_data)
    np.savetxt(train_file,train_data,fmt='%.5f')

    test_data=[]
    for batch in val_loader:
        label, data = batch
        label,data=label.to(cfg.device),data.to(cfg.device)
        logits = model.forward(data)
        feature = logits[1].cpu().numpy().reshape(-1)
        line = np.insert(feature, 0, label.item()).tolist()
        test_data.append(line)
    test_data = np.array(test_data)
    np.savetxt(test_file,test_data, fmt='%.5f')

