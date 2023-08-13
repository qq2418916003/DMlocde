# -*- coding: UTF-8 -*-
# @Date : 2022/4/8 21:07
# @Author : chenshuai
# @File : app.py
import joblib
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from FeatureExtraction import featureExtraction
import pandas as pd
import os


testPath = "./data/test.txt"
featuresPath = "./feature/"
resultPath = "./result/result.xlsx"
resultPath1 = "./result/result1.xlsx"
resultPath2 = "./result/result2.xlsx"
importancesAddress = "./featureSelection/importancesAddress"


GOclass = ['Cytoplasm', 'Endoplasmic', 'Extracellular', 'Mitochondria', 'Nucleus']


def featureVector(fileLine, dim):
    """
    将单个特征数据转换为向量
    :param fileLine:  文件的每一行代表一个特征数据
    :return:   向量
    """
    vector = np.zeros((1, dim))
    label = None
    for i in range(dim):
        label = int(fileLine.split(":")[0].split()[0])
        data = fileLine.split(":")[i + 1].split()[0]
        vector[0, i] = (float(data))
    return vector, label


def extractionSampleNo(dataAddress):
    uniPortKB = []
    records = pd.read_csv(dataAddress, header=None)
    fasta_sequences = []
    for fasta in records.values:
        uniPortKB.append("-")
    # proteinSeqFile = open(dataAddress, "r", encoding="UTF-8")
    # for line in proteinSeqFile:
    #     if line.startswith('>'):
    #         name = line.strip().split('|')[1]
    #         uniPortKB.append(name)
    # proteinSeqFile.close()

    return uniPortKB


def data_process(testFileName, featureDim):
    """
    将特征数据转换为特征向量
    :param testFileName: 测试数据集
    :param featureDim: 特征维度
    :return: 数据特征向量矩阵，数据标签
    """

    index = 0
    testFile = open(testFileName, "r")
    testLines = testFile.readlines()
    testSum = len(testLines)
    testMap = np.zeros((testSum, featureDim))
    data_label = []
    for line in testLines:
        testMap[index, :], label = featureVector(line, featureDim)
        data_label.append(label)
        index += 1
    testFile.close()

    # print("测试数据共有：%d" % (testSum))

    return testMap, data_label


def featureCombination(testPath, feature):
    """
    将不同特征组合起来转化为特征向量
    :param testPath: 测试集地址
    :param feature: 特征
    :return: 组合后的特征向量
    """
    tempTestMap = []
    data_label = None
    for name in feature:
        testFile = testPath + name + '.txt'
        featureDim = feature[name]
        # print('特征{}：'.format(name), end='')

        testMap, data_label = data_process(testFile, featureDim)
        tempTestMap.append(testMap)

    allTestMap = tempTestMap[0]
    for i in range(len(tempTestMap) - 1):
        allTestMap = np.concatenate((allTestMap, tempTestMap[i + 1]), axis=1)

    allTestMap = MinMaxScaler().fit_transform(allTestMap)

    return allTestMap, data_label


def featureSelection(testMap, feature):
    """
    特征选择
    :param testMap: 测试数据
    :return: 降维后的特征向量
    """

    tr = '_'
    feature = tr.join(list(feature))
    if feature == 'AAC_DPC':
        featureDim = 420
        dim = 300
    if feature == 'GTPC_CTriad':
        featureDim = 468
        dim = 350
    if feature == 'SOCNumber_QSOrder':
        featureDim = 160
        dim = 140

    importances = np.zeros(featureDim)
    importancesFile = open(importancesAddress + feature + '.txt', 'r')
    fileLine = importancesFile.readline()
    importancesFile.close()
    line = fileLine.strip()
    for i in range(featureDim):
        data = line.split(',')[i]
        importances[i] = (float(data))
    temp = importances
    temp.sort()
    temp = temp[::-1]
    # print(temp)
    threshold = temp[dim - 1]
    # print(threshold)
    testMap = testMap[:, importances >= threshold]
    featureDim = testMap.shape[1]
    # print("降维后的特征数：" + str(featureDim))
    return testMap


def featuresProcess(testPath, feature):
    """
    特征数据处理
    :param testPath: 测试集
    :param feature: 特征
    :return:
    """

    # 将输入的特征数据转换为特征向量
    testMap, data_label = featureCombination(testPath, feature)

    # 特征选择
    testMap = featureSelection(testMap, feature)

    return testMap, data_label


def model_1(testPath, featuresPath, model, name):
    """
    根据AAC_DPC组合特征和训练好的模型得到预测结果
    :param testPath: 测试数据
    :param featuresPath: 特征
    :return:
    """
    features = ['AAC', 'DPC']
    AAC_DPC = {'AAC': 20, 'DPC': 400}

    # 特征提取
    featureExtraction(testPath, features, featuresPath)

    # 特征数据处理
    testMap, data_label = featuresProcess(featuresPath, AAC_DPC)
    # 训练模型
    # 预测模型
    if name=='lightgbm':
        predict_proba = model.predict(testMap)
        print(predict_proba)
    else:
        predict_proba = model.predict_proba(testMap)
    return predict_proba


def model_2(testPath, featuresPath, model2, name):
    """
    根据GTPC_CTriad组合特征和训练好的模型得到预测结果
    :param testPath: 测试数据
    :param featuresPath: 特征
    :return:
    """
    features = ['GTPC', 'CTriad']
    GTPC_CTriad = {'GTPC': 125, 'CTriad': 343}
    # 特征提取
    featureExtraction(testPath, features, featuresPath)
    # 特征数据处理
    testMap, data_label = featuresProcess(featuresPath, GTPC_CTriad)

    # 预测模型
    if name == 'lightgbm':
        predict_proba = model2.predict(testMap)
    else:
        predict_proba = model2.predict_proba(testMap)
    return predict_proba


def model_3(testPath, featuresPath, model3, name):
    """
    根据SOCNumber_QSOrder组合特征和训练好的模型得到预测结果
    :param testPath: 测试数据
    :param featuresPath: 特征
    :return:
    """

    features = ['SOCNumber', 'QSOrder']
    SOCNumber_QSOrder = {'SOCNumber': 60, 'QSOrder': 100}

    # 特征提取
    featureExtraction(testPath, features, featuresPath)

    # 特征数据处理
    testMap, data_label = featuresProcess(featuresPath, SOCNumber_QSOrder)

    # 预测模型
    if name == 'lightgbm':
        predict_proba = model3.predict(testMap)
    else:
        predict_proba = model3.predict_proba(testMap)

    return predict_proba


def PlantGO(testPath, model1, model2, model3, name):
    """
    预测小麦蛋白质的多种功能
    :param testPath: 测试数据
    :return:
    """
    predict_proba = model_1(testPath, featuresPath, model1)
    predict_proba2 = model_2(testPath, featuresPath, model2)
    predict_proba3 = model_3(testPath, featuresPath, model3)

    allPredict_proba = predict_proba + predict_proba2 + predict_proba3
    allPredict_proba = allPredict_proba / 3
    print(allPredict_proba)
    # 模型1的结果
    predict1 = MinMaxScaler(feature_range=(0, 1)).fit_transform(predict_proba)
    # 模型2的平均结果
    predict2 = MinMaxScaler(feature_range=(0, 1)).fit_transform(predict_proba2)
    # 三个模型的平均结果
    predict3 = MinMaxScaler(feature_range=(0, 1)).fit_transform(predict_proba3)
    # 三个模型的平均结果
    predict = MinMaxScaler(feature_range=(0, 1)).fit_transform(allPredict_proba)
    # predict[predict < 0.5] = 0
    # predict[predict >= 0.5] = 1
    # predict = predict.astype(int)
    # print(predict)

    # result = np.where(predict == 0, "False", "True")
    result = predict
    # print(result)
    print("测试集路径:", testPath)
    # uniPortKB = extractionSampleNo(testPath)
    # print(uniPortKB)
    df = pd.DataFrame(result, columns=GOclass)
    df.to_excel(resultPath)
    df1 = pd.DataFrame(predict1, columns=GOclass)
    df1.to_excel(resultPath1)
    df2 = pd.DataFrame(predict2, columns=GOclass)
    df.to_excel(resultPath2)
    df3 = pd.DataFrame(predict3, columns=GOclass)
    resultPath3 = "./result/"+name+"_result3.xlsx"
    df3.to_excel(resultPath3)
    print("Finish")


'''if __name__ == '__main__':
    name = input("请输入所需要的分类器进行预测(例如:KNN, SVM):")
    model = joblib.load(name + '_Model_1.pkl')
    model2 = joblib.load(name + '_Model_2.pkl')
    model3 = joblib.load(name + '_Model_3.pkl')
    PlantGO(testPath, model, model2, model3, name)'''











