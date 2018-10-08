# -*- coding:utf-8 -*-
"""
数据集划分为2个部分
    训练集  70% 80% 75%
    测试集  30% 20% 25%



"""
from sklearn.datasets import load_iris, load_boston, fetch_20newsgroups

# 都是返回datasets.base.Bunch(类字典格式)
# data: 特征数据数组,是 numpy.ndarray的二维数组
# target: 标签数组, 是 numpy.ndarray的一维数组
# DESCR: 数据描述
# feature_names: 特征名,新闻数据,手写数字,回归数据集没有feature_names
# target_names: 标签名
"""
sklearn.datasets

    加载获取流行数据集
    datasets.load_*()
        获取小规模数据集，数据包含在datasets里
    datasets.fetch_*(data_home=None)
        获取大规模数据集，需要从网络上下载，函数的第一个参数是data_home，表示数据集下载的目录,默认是 ~/scikit_learn_data/

"""

""" 分类数据集 """
lr = load_iris()  #
print("特征值：", lr.data)
print("目标值：", lr.target)
print("描述：", lr.DESCR)
print("feature_names：", lr.feature_names)
print("target_names：", lr.target_names)

""" 回归数据集 """
# lr = load_boston()
# print("特征值：", lr.data)
# print("目标值：", lr.target)
# print("描述：", lr.DESCR)
# print("target_names：", lr.target_names)


# 数据量比较大的数据集
# lr = fetch_20newsgroups(subset='all')  # data_home 下载目录  subset=all 获取所有
# print("特征值：", lr.data)
# print("目标值：", lr.target)
# print("描述：", lr.DESCR)
# print("target_names：", lr.target_names)  # 20个类别代表的字符串,索引下标

""" 数据集的划分接口 """
# sklearn.model_selection.train_test_split(arrays, *options)
# 1. x数据集的特征
# 2. y数据集的标签值
# 3. test_size 测试集的大小 一般为float
# 4. random_stats 随机数种子,不同的种子会造成不同的采样结果
# 5. return  默认4个返回值  先都是特征值
#       1.训练集特征值
#       2.测试集特征值
#       3.训练标签值(目标值)
#       4.测试标签值(目标值)(默认随机取)

# 数据集的训练集和测试集划分
from sklearn.model_selection import train_test_split

# 假设 x,y 代表特征值,目标值, train,test 代表训练集,测试集
# 特征值,   目标值(一一对应),
#
# x_train, x_test, y_train, y_test = train_test_split(lr.data, lr.target, test_size=0.3)
# practice_data, test_data, practice_target, test_target = train_test_split(lr.data, lr.target, test_size=0.3)
#
# print("训练集的特征是:", practice_data, practice_data.shape)
# print("*" * 100)
# print("测试集的特征是:", test_data, test_data.shape)
# print("*" * 100)
# print("训练集的目标值:", practice_target, practice_target.shape)
# print("*" * 100)
# print("测试集的目标值:", test_target, test_target.shape)


# fit(a) 就是计算每一列平均值和标准差
# transform(a) 谁调用它 ==>谁使用自己的平均值和标准差去转换(a)
# fit_transform(a)  #  使用自己的平均值和标准差去转换


# estimator 估计器

