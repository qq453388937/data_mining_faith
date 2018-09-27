# -*- coding:utf-8 -*-

# 特征预处理  [数值型数据]   --->  通过一些转换函数将特征数据转换成更加适合算法模型的特征数据过程
# 缺失值处理  基本数据处理等pandas来做

# 为什么我们要进行归一化/标准化？
# 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征

# 数值型数据->无量纲化(不同数据的大小,方差不一样): 归一化,标准化  容易影响目标结果的处理  preprocessing 里面

# 无量纲化一句话: 不同规格的数据转换到统一规格

# 特征的单位或者大小相差较大，或者某特征的方差相比其他的特征要大出几个数量级，容易影响（支配）目标结果，使得一些算法无法学习到其它的特征

# 归一化 [0,1] X`= (x - min)/(max - min) X``= X`*(mx-mi)+mi  mx=1 mi=0  通过对原始数据进行变换把数据映射到[0,1] 之间

# 归一化容易受到异常大值和异常小值影响, 这种方法鲁棒性较差,只适合传统精确小数据场景

# 标准化: 对原始数据进行变换到均值为0 标准差(方差求根号)为1范围内  抵抗异常值的影响!!!!!
# 公式 X` (x - mean) / thta
# (x1-mean)^2+(x2-mean)^2+.../n  --> 方差0 同1数据  方差越大, 数据越来越离散
"""
对于归一化来说：如果出现异常点，影响了最大值和最小值，那么结果显然会发生改变
对于标准化来说：如果出现异常点，由于具有一定数据量(不受影响的前提)，少量的异常点对于平均值的影响并不大，从而方差改变较小。
"""

"""
降维: 降维 -> 降低特征的数量 得到一些不相关的特征
eg: 湿度与降雨量的关系 ，十分相关

特征选择 的方法：
    Filter(过滤式)：主要探究特征本身特点、特征与特征之间关联
        低方差特征过滤   方差选择法
        相关系数    皮尔逊相关系数(衡量关系密切程度)  --> [-1,1]    
        相关特征必须做处理: 删掉 or 合成
    Embedded(嵌入式)：算法自动选择特征（特征与目标之间的关联）
        决策树： 信息嫡，信息增益
        正则化： L1,L2
        深度学习： 卷积等

 特征降维：主成分分析(PCA) ：压缩，删除冗余，可以创造新的特征
            定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量
            作用：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。
            应用：回归分析或者聚类分析当中
         
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
    """ 使用低方差方法过滤特征(股票指标过滤) """

    factor = pd.read_csv("factor_returns.csv")
    # 使用VarianceThreshold , 9列指标低方差过滤
    var = VarianceThreshold(threshold=1)
    data = var.fit_transform(factor.iloc[:, 1:10])
    print(data)
    print(data.shape)  # 看过滤了哪一列 ,缺点 不好判断方差大小


def relations_func_personer():
    """
        皮尔逊相关系数
            相关系数的值介于–1与+1之间，即–1≤ r ≤+1。其性质如下：

            当r>0时，表示两变量正相关，r<0时，两变量为负相关
            当|r|=1时，表示两变量为完全相关
            当r=0时，表示两变量间无相关关系
            当0<|r|<1时，表示两变量存在一定程度的相关。且|r|越接近1，两变量间线性关系越密切；|r|越接近于0，表示两变量的线性相关越弱
            一般可按三级划分：|r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关

    """
    from scipy.stats import pearsonr
    import tushare as ts

    # print(tushare.__version__)  # 1.2.12
    stock_data = ts.get_hist_data("600015")
    stock = stock_data
    col = ["open", "high", "close", "v_ma5", "v_ma10", "v_ma20"]
    for i in range(len(col)):
        for j in range(i, len(col) - 1):
            print("指标%s和%s 相关系数计算==>%f" % (
                col[i], col[j + 1], pearsonr(stock[col[i]], stock[col[j + 1]])[0]
            ))
    # print(stock)


from sklearn.decomposition import PCA


def PCA_demo():
    """
     特征降维：主成分分析(PCA) ：压缩，删除冗余，可以创造新的特征
            定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量
            作用：是数据维数压缩，尽可能降低原数据的维数（复杂度），损失少量信息。
            应用：回归分析或者聚类分析当中
    """
    import tushare as ts

    # a = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    # pca = PCA(n_components=0.95)  # 整数表示保留几个信息，小数表示百分之多少 一般大于95%
    # data = pca.fit_transform(a)
    # print(stock)

    # 2个合成1个
    stock_data = ts.get_hist_data("600015")
    pca = PCA(n_components=1)  # 整数表示保留几个信息，小数表示百分之多少 一般大于95%
    stock = stock_data
    data = pca.fit_transform(stock[['open']], stock[['close']])  # 举例列不是实际的列
    print(data)
    pass


def analysize():
    # 导入4个表
    prior = pd.read_csv("./instacart/order_products__prior.csv")
    products = pd.read_csv("./instacart/products.csv")
    orders = pd.read_csv("./instacart/orders.csv")
    aisles = pd.read_csv("./instacart/aisles.csv")

    # 表合并       on --> 两张表共同拥有的键,默认内连接
    a1 = pd.merge(prior, products, on="product_id")
    a2 = pd.merge(a1, orders, on="order_id")
    ret = pd.merge(a2, aisles, on="aisle_id")
    # pd.crosstab 统计用户与物品类别之间的次数关系（统计次数）
    cs_ret = pd.crosstab(ret['user_id'], ret['aisle'])
    print(cs_ret)
    print(cs_ret.shape)
    pca = PCA(n_components=0.95)  # 整数表示保留几个信息，小数表示百分之多少 一般大于95%
    res_final = pca.fit_transform(cs_ret)

    index = ["第%s个分类" % i for i in range(44)]
    index2 = ["第%s个用户" % i for i in range(len(res_final))]
    res_pd = pd.DataFrame(res_final, index=index2, columns=index)
    print(res_pd)
    pass


if __name__ == '__main__':

    # minmaxscaler()
    # stardscaler()
    # relations_func_personer()
    # PCA_demo()
    analysize()
    pass
