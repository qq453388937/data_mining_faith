# -*- coding:utf-8 -*-

# 缺失值处理  基本数据处理等pandas来做

# 特征预处理  通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程   ---> 数值型数据

# 无量纲化(不同数据的大小,方差不一样): 归一化,标准化  容易影响目标结果的处理  preprocessing 里面

# 一句话: 不同规格的数据转换到统一规格

# 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征

# 归一化 [0,1]  X`*(mx-mi)+mi  mx=1 mi=0  通过对原始数据进行变换把数据映射到[0,1] 之间

# 归一化容易受到异常大值和异常小值影响, 这种方法鲁棒性较差,只适合传统精确小数据场景

# 标准化: 对原始数据进行变换到均值为0 标准差(方差求根号)为1范围内  抵抗异常值的影响!!!!!
# 公式 X` (x - mean) / thta
# (x1-mean)^2+(x2-mean)^2+.../n  --> 方差0 同1数据  方差越大, 数据越来越离散
"""
对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变
对于标准化来说：如果出现异常点，由于具有一定数据量(不受影响的前提)，少量的异常点对于平均值的影响并不大，从而方差改变较小。
"""

"""
降维: 维-> 降低特征的数量 得到一些不相关的特征
eg: 湿度与降雨量的关系

特征选择
    Filter(过滤式)：主要探究特征本身特点、特征与特征之间关联
    低方差特征过滤   方差选择
    相关系数    皮尔逊相关系数(衡量关系密切程度)  --> [-1,1]    
    相关特征必须做处理: 删掉 or 合成
"""
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


def minmaxscaler():
    """ 对约会对象数据进行归一化处理 """
    # 读取数据
    dating = pd.read_csv('dating.txt')
    # dating = dating.sort_index()
    # print(dating)
    # print(dating.index)
    # print(dating.columns)
    res = dating[['milage', 'Liters', 'Consumtime']]
    # 实例化minmaxscaler 进行 fit_transform
    mm = MinMaxScaler(feature_range=(0, 1))  # feture_range 默认 0 到 1  可以不传
    print(res)
    new_data = mm.fit_transform(res)  # fit
    print(new_data)
    pass


def stardscaler():
    """ 标准化后处理的每列特征数据都聚集在均值为0附近,标准差为1 的范围"""
    dating = pd.read_csv("dating.txt")
    res = dating[["milage", "Liters", "Consumtime"]]
    sd = StandardScaler()
    print(res)
    new_data = sd.fit_transform(res)
    print(new_data)
    pass


from sklearn.feature_selection import VarianceThreshold


def varthreshold():
    """ 使用地方差方法过滤特征(股票指标过滤) """

    factor = pd.read_csv("factor_returns.csv")
    # 使用VarianceThreshold , 9列指标低方差过滤
    var = VarianceThreshold(threshold=1)
    data = var.fit_transform(factor.iloc[:, 1:10])
    print(data)
    print(data.shape)  # 看过滤了哪一列 ,缺点 不好判断方差大小


from scipy.stats import pearsonr

if __name__ == '__main__':

    # minmaxscaler()
    stardscaler()
