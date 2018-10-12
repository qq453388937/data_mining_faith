# -*- coding:utf-8 -*-
"""
线性回归(Linear regression)是利用回归方程(函数)对一个或多个自变量(特征值)和因变量(目标值)之间关系进行建模的一种分析方式。

我们看到特征值与目标值之间建立的一个关系，这个可以理解为回归方程。
y = wx + b (偏置,截距)
    权重

    损失函数, 优化权重和偏正
    目的是找到最小损失的对应的W值
    最小二乘法

    正规方程: 直接通过特征值和目标值之间的矩阵运算得出

    梯度下降
           学习速率和方向


"""

from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score  # 均方误差评估回归性能
from sklearn.externals import joblib
from sklearn.cluster import KMeans, k_means


def xian_xing_hui_gui():
    """
        线性回归进行房价预测
        高次项回归
        岭回归 = 线性回归 + L2正则化
        LASSO回归 = 线性回归 + L1正则化
        决策树-剪枝, 正则化
        正则化系数越大,权重越小


    """
    # 1. 获取数据并且进行分割
    data = load_boston()
    """ 目标值也可以标准化 """
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)
    print(x_train)
    # 2. 标准化处理, 和权重相乘有影响 (线性回归)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)  # 训练集特征值标准化

    x_test = std.fit_transform(x_test)  # 测试集特征值标准化
    # # 正规方程求解方式
    # lr = LinearRegression()   #  or 岭回归
    # lr.fit(x_train, y_train)  # fit后已经有参数结果
    # print("正规方程计算出的权重:", lr.coef_)
    # print("正规方程计算出的偏置:", lr.intercept_)
    # result = lr.predict(x_test)  # 预测目标值,只需传入测试集
    # print(result)
    # print(y_test)

    # # 判定回归性能评估
    # # 均方误差
    # error = mean_squared_error(y_test, result)
    # print("正规方程方式误差:", error)
    #
    # # 梯度下降的方法进行预测
    #
    # # sgd = SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate="invscaling")
    # sgd = SGDRegressor(loss="squared_loss", fit_intercept=True, learning_rate="constant", eta0=0.5)
    # sgd.fit(x_train, y_train)
    # result2 = sgd.predict(x_test)
    #
    # error2 = mean_squared_error(y_test, result2)
    # print("梯度下降方式误差:", error2)

    rd = Ridge()
    rd.fit(x_train, y_train)
    print("lin回归的权重:", rd.coef_)
    print("lin回归计算出的偏置:", rd.intercept_)
    res = rd.predict(x_test)
    # # 均方误差
    error = mean_squared_error(y_test, res)
    print("lin回归方式误差:", error)
    #


import pandas as pd
import numpy as np


def luo_ji_hui_gui():
    """ 使用逻辑回归进行预测 """
    # 读取数据处理缺失值
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin"
        ".data", names=column_name)
    print(data.shape)
    # 替换? 为 np.nan 并且重新赋值
    data = data.replace(to_replace="?", value=np.nan)
    data = data.dropna()
    print(data.shape)
    print(data)

    # 分割数据训练集,测试集
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:10], data.iloc[:, 10], test_size=0.3)

    # 进行标准化(前面的输入也是一个线性回归)

    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    # 进行逻辑回归预测
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    # print("逻辑回归的权重:", lr.coef_)
    # print("逻辑回归计算出的偏置:", lr.intercept_)
    res = lr.predict(x_test)
    # print(res)
    # print(y_test)
    # print("逻辑回归准确率:", lr.score(x_test, y_test))
    # print("精确率和召回率为：", classification_report(y_test, lr.predict(x_test), labels=[2, 4], target_names=['良性', '恶性']))

    # y_test = np.where(y_test > 2.5, 1, 0)  # 必须以 1 和 0 标记
    #
    # print("此场景分类器的AUC指标为", roc_auc_score(y_test, res))
    joblib.dump(lr, 'lr.pkl')
    model = joblib.load("lr.pkl")
    res = model.predict(x_test)
    print(res)
    print(y_test)
    print("逻辑回归准确率:", lr.score(x_test, y_test))




if __name__ == '__main__':
    # xian_xing_hui_gui()
    luo_ji_hui_gui()
