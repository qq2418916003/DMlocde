import torch
from torch.utils.data import Dataset, DataLoader
# import jieba
import numpy as np


def read_voc_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    for item in dict_list:
        item=item.split(',')
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict


def load_data(data_path,stop_word_path):
    datalist = open(data_path, 'r').readlines()  # 文本行列表
    # 获取停用词列表
    # stopwordlist = open(stop_word_path).readlines()  # 停用词列表
    # stopwordlist = [x.strip() for x in stopwordlist]
    # stopwordlist.append(' ')
    # stopwordlist.append('\n')
    max_seq_len = 0
    # voc_dict = {}
    data = []
    for index, item in enumerate(datalist):
        info = item.split(',')
        label = info[0]
        content = info[1].strip()
        seg_res = []  # 保存过滤掉停用词后的每句分词后的列表
        # seg_list = jieba.lcut(content, cut_all=False)  # 结巴分词
        seg_list = list(content)
        # seg_list = []
        # step = 3
        # for i in range(0, len(content) - 1, step):
        #     seg_list.append(content[i:i + step])
        # print(index)
        for seg_item in seg_list:
            # if seg_item in stopwordlist:
            #     continue
            seg_res.append(seg_item)
            # # 给词频词典添加统计信息，词:出现数量
            # if seg_item in voc_dict.keys():
            #     voc_dict[seg_item] += 1
            # else:
            #     voc_dict[seg_item] = 1
        if len(seg_res)>max_seq_len:
            max_seq_len=len(seg_res)
        data.append([label,seg_res])
    return data,max_seq_len


class MyDataSet(Dataset):
    def __init__(self, voc_dict_path, data_path, stop_word_path):
        self.voc_dict = read_voc_dict(voc_dict_path)
        self.data, self.max_sql_len = load_data(data_path, stop_word_path)
        self.max_sql_len = 1000
        # np.random.shuffle(self.data)

    def __getitem__(self, index):
        data=self.data[index]
        label=int(data[0])
        word_list=data[1]
        input_idx=[]
        for index,word in enumerate(word_list):
            if index>(self.max_sql_len-1):
                continue
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict['<UNK>'])
        if len(input_idx) < self.max_sql_len:
            input_idx = input_idx+[self.voc_dict['<PAD>'] for _ in range(self.max_sql_len-len(input_idx))]
        return torch.tensor(label, dtype=torch.int64), torch.tensor(np.array(input_idx))

    def __len__(self):
        return len(self.data)


# if __name__ == '__main__':
#     dataset = MyDataSet('data/dict', 'data/train.txt', 'data/hit_stopwords.txt')
#     loader = DataLoader(dataset, batch_size=10)
#     for i, item in enumerate(loader):
#         print(i)