import torch
import torch.nn as nn

def Feature_Clustering(feature_vectors,per_mean,all_mean):
    """
    feature_vectors表示数据维度：[100,512]
    第一维表示类别数，第二维表示每一类中的样本数，第三维表示数据维度
    per_mean表示每个类别特征的均值：[10,512]
    all_mean表示所有类别特征的均值：[1,512]
    """
    sample_num = feature_vectors.shape[0]
    class_num = per_mean.shape[0]
    sample_per_class = sample_num / class_num
    d1 = (feature_vectors - per_mean.repeat(int(sample_per_class), 1)).norm(p=2,dim=1)#100
    
    d2 = (per_mean-all_mean).norm(p=2,dim=1)#10

    R_fc = (class_num/sample_per_class)*sum(d1*d1)/sum(d2*d2)
    return R_fc



