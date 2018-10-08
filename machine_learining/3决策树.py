# -*- coding:utf-8 -*-

"""
    决策树 每个类别的概率 乘以 log 每个类别的概率
    H = -(1/32 * log 1/32 + ...) = -log 1/32 = 5
    信息熵: 信息和消除不确定性是相联系的
    信息增益: 得知某个特征对总的信息熵减少的大小
             减少越大,可以放在树的顶部
    H(D) = -(6/15*log(6/15)+9/15*log(9/15))  = 0.971
    g(D,A) = H(D) - H(D|A)
    计算信息增益:
    g(D,年龄) = 0.971 -[5/15H(青年)+5/15H(中年)+5/15H(老年)] = 0.313
            H(青年) = -(3/5log(3/5)+2/5(log(2/5)))
            H(中年) = -(3/5log(3/5)+2/5(log(2/5)))
            H(老年) = -(4/5log(4/5)+1/5(log(1/5)))
    g(D,有工作) =  0.324
    g(D,有自己房子) =  0.420
    g(D,贷款情况) = 0.363
"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


def decision_tree():
    """ 决策树预测乘客分类(过拟合) """
    # 1. 获取乘客数据
    data = pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    # 2. 找出特征值,目标值
    x = data[['pclass', 'sex', 'age']]
    y = data['survived']
    # 3. 缺失值处理, 特征类别数据处理 --> one-hot 编码
    x['age'].fillna(x['age'].mean(), inplace=True)
    dv = DictVectorizer(sparse=False)
    # [["1st","2","female"],[]]--->[{"pclass":, "age":2, "sex: female"}, ]
    x = dv.fit_transform(x.to_dict(orient="records"))  # --> one-hot 编码
    print(dv.get_feature_names())
    print(x)
    # 3.5 分割数据集 可分可不分
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # # 4.决策树接口预测生存分类
    ds = DecisionTreeClassifier()
    ds.fit(x_train, y_train)
    print(ds.predict(x))
    print("预测测试集当中的结果:", ds.predict(x)[:50])  # 预测测试集的文档类别,用测试集来分类
    print("测试集当中的真实结果", y_test[:50])
    print("决策树预测的准确率", ds.score(x_test, y_test))
    # pass
    # export_graphviz(ds, out_file="./tree.dot",
    #                 feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', 'sex=female', '男性'])
    # 4 决策森林
    rf = RandomForestClassifier()

    # 构造超参数字典
    param = {
        "n_estimators": [1, 3, 5, 7, 10],
        "max_depth": [5, 8, 12, 15],
        "min_samples_split": [2, 3, 5]
    }
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)

    print("随机森林的准确率", gc.score(x_test, y_test))
    print("交叉验证的结果最佳参数", gc.best_estimator_)


if __name__ == '__main__':
    decision_tree()
