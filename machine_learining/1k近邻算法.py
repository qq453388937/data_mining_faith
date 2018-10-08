# -*- coding:utf-8 -*-
# K-近邻算法定义
# 如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
# K值影响 算法结果
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV


def kn():
    """
        k近邻算法预测用户查询业务  (惰性,不是学习的算法)
        k值取很小: 容易受异常点影响
        k值取很大: 受到样本均衡的问题
        0<x<10
        0<y<10
        1. 缩小数据范围(上课节约时间，非必要，生产不做)

    """
    data = pd.read_csv("./1k/train.csv")
    # 1.缩小数据范围,非必须
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")

    # print(data)

    # # 3。将签到位置小于 n 个人的位置给删除掉
    place_count = data.groupby('place_id').count()  # group_by 列为索引列
    tf = place_count[place_count.row_id > 3]
    # print(tf)
    # print(tf.index)
    tf = place_count[place_count.row_id > 3].reset_index()  # 重置索引列为数字自增
    data = data[data['place_id'].isin(tf.place_id)]

    # print(tf)
    # print(tf.index)
    # # 4. 数据划分
    # # 所有数据特征值
    data_use = data[["x", "y", "accuracy", "time"]]
    # # 所有数据目标值
    target = data["place_id"]  # 2个[[]] 是二维数组 , 正常是先列后行[""] or 点....
    # print(data)
    # print("*************************************")
    # print(target)
    # print(target.shape)
    # print(target.index)
    # print("*************************************")

    # 训练集特征值    测试集特征值    训练目标值       测试目标值
    practice_data, test_data, practice_target, test_target = train_test_split(data_use, target,
                                                                              test_size=0.3)  # 30% 测试集
    # # 2.标准化  (训练集特征值和测试集特征值做标准化处理提高预测成功率)
    # 2.1 对训练集标准化
    sd = StandardScaler()
    practice_data = sd.fit_transform(practice_data)
    # 2.2 对测试集特征值标准化
    test_data = sd.fit_transform(test_data)

    # # # 5.使用 k 近邻算法
    # knn = KNeighborsClassifier(n_neighbors=5)  # 默认5可不传
    # #
    # # print(practice_target.shape)
    # # # 6.fit 训练  --> 估计器流程1
    # knn.fit(practice_data, practice_target)  # 传入训练集数据,训练集目标值
    # #
    # # # 7.预测测试集 --> 估计器流程2
    # y_predict = knn.predict(test_data)
    # print("预测结果", y_predict)  # 打印预测结果
    # #
    # # # 8.准确率
    # print("k 近邻算法的准确率为:", knn.score(test_data, test_target))

    # 交叉验证: 为了让被评估的模型更加准确可信, 为了看一个参数在这个数据集当中综合表现情况
    # 通常:十折交叉认证
    # 超参数: 需要手动调整参数  通常是超参数搜索-网格搜索 + 交叉验证 对 k 近邻算法进行调优

    knn = KNeighborsClassifier()  # 默认5可不传
    # 构造超参数字典
    param = {"n_neighbors": [1, 3, 5, 7, 10]}

    # 测试2折,通常10折
    gc = GridSearchCV(knn, param_grid=param, cv=2)
    # fit 输入数据
    gc.fit(practice_data, practice_target)
    print(gc.predict)
    # 查看模型超参数调优的过程和交叉验证的结果
    print("在2折交叉验证中最好的结果:", gc.best_score_)
    print("选择最好的模型参数:", gc.best_estimator_)
    print("每次交叉验证,验证集的表现结果:", gc.cv_results_)
    print("在测试集当中的最终预测表现", gc.score(test_data, test_target))


if __name__ == '__main__':
    kn()
    pass
