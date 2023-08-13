# -*- coding: utf-8 -*-
# @Time : 2021/7/29 9:12
# @Author : chenshuai
# @File : FeatureExtraction.py

import os

features = ["AAC", "CTriad", "DPC", "QSOrder", "GTPC"]


def command(dataAddress, feature, featureAddress):
    """
    通过cmd命令执行脚本文件提取特征
    :param dataAddress: 输入：数据集地址
    :param feature: 输入：特征
    :param featureAddress: 输出：特征向量地址
    :return: 提取特征的命令
    """

    return "python featureExtraction/feature.py --file " + dataAddress + " --method " + feature + " " \
                "--format svm --out " + featureAddress + feature + ".txt"


def execute_cmd_command(command):
    return os.popen(command).read()


def featureExtraction(dataAddress, features, featureAddress):
    """
    提取特征
    :param dataAddress: 输入：数据集地址
    :param features: 输入：特征
    :param featureAddress: 输出：特征集保存地址
    """

    for feature in features:
        result = execute_cmd_command(command(dataAddress, feature, featureAddress))


def kflodData_process(dataAddress, features, featureAddress):
    """
    提取10折交叉验证数据集的特征
    :param dataAddress: 输入：10折数据集地址
    :param features: 输入：特征
    :param featureAddress: 输出：特征集保存地址
    :return:
    """

    for i in range(1, 11):
        trainDataAddress = dataAddress + "train" + str(i) + ".fasta"
        testDataAddress = dataAddress + "test" + str(i) + ".fasta"
        trainFeatureAddress = featureAddress + "train" + str(i) + "\\"
        testFeatureAddress = featureAddress + "test" + str(i) + "\\"

        featureExtraction(trainDataAddress, features, trainFeatureAddress)
        featureExtraction(testDataAddress, features, testFeatureAddress)


if __name__ == '__main__':
    print('1')

