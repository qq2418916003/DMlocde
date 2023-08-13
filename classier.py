# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 20:40
# @Author  : @HelloWord_Naa
# @FileName: classier.py
# @Software: PyCharm
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from config import Config
from mydataset import MyDataSet

svm = SVC(kernel="rbf", C=100, gamma=0.01)
cfg = Config()

train_dataset = MyDataSet(voc_dict_path='data/dict', data_path='data/train.txt', stop_word_path='data/hit_stopwords.txt')
cfg.pad_size = train_dataset.max_sql_len
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
for epoch in range(cfg.epochs):
    for index, batch in enumerate(train_loader):
        label, data = batch
        svm.fit(data, label)

# y_predict = svm.predict(x_test_pca)

