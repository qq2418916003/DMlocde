#此脚本用于创建词频反向排序字典，用于之后embedding编码
#获取文本内容
datapath='data/train.txt'
data_stop_path='data/hit_stopwords.txt' #哈工大停用词表
datalist=open(datapath,'r').readlines() #文本行列表
#获取停用词列表,对非中文语言无效
stopwordlist=open(data_stop_path).readlines()#停用词列表
stopwordlist=[x.strip() for x  in stopwordlist]
stopwordlist.append(' ')
stopwordlist.append('\n')
#定义词频词典
voc_dict=dict()
min_seq=1 #最小出现数量，出现次数小于该数量的词被忽略
top_n=1000#词频词典最后只保留这些数量的词
UNK='<UNK>'
PAD='<PAD>'
count=0
for item in datalist:
    count+=1
    print(count)
    # info=item.split('\t')
    info=item.split(',')
    label=info[0].strip()
    content=info[1].strip()
    seg_res=[] #保存过滤掉停用词后的每句分词后的列表
    seg_list=list(content) #单字母作为词，根据预料内容将单个语料分割为列表，元素为单词
    # seg_list=[]  #下面4行是指定多个字符作为词
    # step = 3
    # for i in range(0, len(content) - 1, step):
    #     seg_list.append(content[i:i + step])

    for seg_item in seg_list:
        if seg_item in stopwordlist:
            continue
        seg_res.append(seg_item)
        #给词频词典添加统计信息，词:出现数量
        if seg_item in voc_dict.keys():
            voc_dict[seg_item]+=1
        else:
            voc_dict[seg_item]=1
#从词频字典中获取降序后的词频列表，取前top个，再从这个列表中获取字典，key是word，value是出现在词频列表中的索引，故value越小出现频率越高
voc_list=sorted([x for x in voc_dict.items() if x[1]>min_seq],key=lambda t:t[1],reverse=True)
voc_list=voc_list[:top_n]
voc_dict={word_count[0]:idx for idx,word_count in enumerate(voc_list)}
voc_dict.update({UNK:len(voc_dict),PAD:len(voc_dict)+1}) #添加空词和填充词
#写入文件
ff=open('data/dict','w')
for k in voc_dict.keys():
    ff.writelines("{},{}\n".format(k,voc_dict[k]))
ff.close()