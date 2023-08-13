from imblearn.over_sampling import SMOTE
import numpy as np
#使用说明，此方法试验后对本数据集无明显效果
# xtrain,numpy数组，shape为[m,n]，不包含标签，
# ytrain,numpy数组，shape为[m,]，值为分类标签，标签是数字如0，1，2，3，4，5
#通过设置参数sampling_strategy字典，指定需要扩充的标签及数量，会返回扩充后的新数据数组及新标签数组
smo=SMOTE(sampling_strategy={3:6000})
xtrain_new,ytrain_new=smo.fit_resample(xtrain,ytrain)

if __name__=='__main__':
    # smote使用示例
    smo_ratio=2 #要求的多数类样本数是少数类样本数的比例
    label_num_dict={1:2500,0:10}
    xtrain=np.random.randint(1,100,5020).reshape(-1,2)
    ytrain=[1]*2500+[0]*10
    mino=label_num_dict[0]
    majo=label_num_dict[1]
    imblance_ratio=majo/mino#多数类是少数类的样本倍数
    print(imblance_ratio)
    if imblance_ratio>smo_ratio:
        smo=SMOTE(sampling_strategy={0:int(majo/smo_ratio)})
        xtrain_new,ytrain_new=smo.fit_resample(xtrain,ytrain)
        print('sum 1',sum(ytrain))
        print('sum 0',len(ytrain_new)-sum(ytrain_new))