# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 1:24
# @Author  : @HelloWord_Naa
# @FileName: model_merge.py
# @Software: PyCharm
import joblib
import torch
from torch.utils.data import DataLoader


from classier_Predict import model_1, model_3, model_2
from model import Confus
from config import Config
from confus_predict import model
from fasttext_predict import cfg
from mydataset import MyDataSet
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

testPath = "./data/test3.txt"
featuresPath = "./feature/"


def model4():
    val_dataset = MyDataSet(voc_dict_path='data/dict', data_path='data/test3.txt',
                                stop_word_path='data/hit_stopwords.txt')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    pre_proba = []
    true_label = []
    model = Confus(Config())
    model.load_state_dict(torch.load('checkpoints/Confus.pth', map_location='cpu'))
    with torch.no_grad():
        for batch in val_loader:
            label, data = batch
            label, data = label.to(cfg.device), data.to(cfg.device)
            logits = model.forward(data)
            pre_proba.append(logits[0][0].numpy())
            true_label.append(label.item())
    return pre_proba, true_label


def ACC(Y_test, Y_pred, n):
    acc = []
    con_mat = confusion_matrix(Y_test, Y_pred)
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        acc1 = (tp + tn) / number
        acc.append(acc1)

    return acc


def merge(model, model2, model3, name):
    predict_proba = model_1(testPath, featuresPath, model, name)
    predict_proba2 = model_2(testPath, featuresPath, model2, name)
    predict_proba3 = model_3(testPath, featuresPath, model3, name)
    predict_proba4, true_label4 = model4()
    predict_proba4 = np.array(predict_proba4)
    allPredict_proba = predict_proba
    # allPredict_proba = predict_proba2
    # allPredict_proba = predict_proba3
    # allPredict_proba = predict_proba4
    # allPredict_proba = predict_proba + predict_proba2 + predict_proba3 + predict_proba4
    # allPredict_proba = allPredict_proba / 4
    # print(allPredict_proba)
    # truearr=allPredict_proba.argmax(axis=-1)==true_label4
    # np.savetxt('data/truearr',truearr)
    print("测试集评估:")
    print("混淆矩阵")
    print(confusion_matrix(true_label4, allPredict_proba.argmax(axis=-1)))
    print(classification_report(true_label4, allPredict_proba.argmax(axis=-1), output_dict=True))
    print(ACC(true_label4, allPredict_proba.argmax(axis=-1), 5))


if __name__ == '__main__':
    name = input("请输入所需要的分类器进行预测(例如:KNN, SVM, lightgbm):")
    model = joblib.load(name + '_Model_1.pkl')
    model2 = joblib.load(name + '_Model_2.pkl')
    model3 = joblib.load(name + '_Model_3.pkl')
    merge(model, model2, model3, name)