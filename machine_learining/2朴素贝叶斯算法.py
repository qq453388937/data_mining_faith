# -*- coding:utf-8 -*-
# 朴素贝叶斯  ---> 文本分类
# 主要就是归为哪个类别的可能性大小,分类思想
"""
联合概率：包含多个条件，且所有条件同时成立的概率
    注意：此条件概率的成立，是由于A1,A2相互独立的结果(记忆),特征相互独立,互不影响
    记作：P(A,B)
    特性：P(A, B) = P(A)P(B)
条件概率：就是事件A在另外一个事件B已经发生条件下的发生概率
    记作：P(A|B)
    特性：P(A1,A2|B) = P(A1|B)P(A2|B)
p(喜欢) = 4/7
p(程序员, 匀称) =  P(程序员)P(匀称) =3/7*(4/7) = 12/49    --> 联合概率,包含多个条件 , 且所有条件同时成立的概率
p(程序员|喜欢) = 1/2    --> 条件概率: 就是事件A在另外一个事件B已经发生条件下的发生概率
P(产品, 超重|喜欢) = P(产品|喜欢)P(超重|喜欢)= 1/2*1/4 = 1/8  -->  综合案例

朴素贝叶斯:
  P(科技|文章) = P(科技|科技词1,科技词2,科技词3.....)
  P(娱乐|文章) = P(娱乐|娱乐词1,娱乐词2,....)
  但是这个公式怎么求？前面并没有参考例子，其实是相似的，我们可以使用贝叶斯公式去计算

  : P(C|W) = P(W|C)P(C)  /  P(W)
  多个条件下一个结果  一个条件下多个结果
  : P(C|F1,F2..) = P(F1,F2..|C)P(C)  / P(F1,F2..)
  由于条件概率特性：P(A1,A2|B) = P(A1|B)P(A2|B) 化解朴素贝叶斯公式
  : P(C|F1,F2..) = P(F1|C)*P(F2|C)*...P(C) / P(F1)*P(F2)*P(F3)
  ---> P(F1│C)=Ni/N （训练文档中去计算）
    计算方法：P(F1│C)=Ni/N （训练文档中去计算）
    拉普拉斯平滑系数:  --> 防止计算出的分类概率为 0
          Ni+a/N+am   Ni+1/N+1*4 --> 4个特征词
    Ni为该F1词在C类别所有文档中出现的次数
    N为所属类别C下的文档所有词出现的次数和

  P(科技|文章1) = P(科技|文章1)P(科技) / P(文章1)  分母可省略比较
  P(娱乐|文章1) = P(娱乐|文章1)P(娱乐) / P(文章1)  分母可省略比较

  例子
     P(科技|被预测文档) = P(科技|影院,支付宝,云计算) => 调用贝叶斯公式省略分母 --> P(影院,支付宝,云计算|科技) * P(科技) --> 分子
                                                                      --> P(影院|科技)*P(支付宝|科技)*P(云计算|科技) * P(科技) --> 分子
                                                                      --> 8/100 * 20/100 * 63/100 * 30/90
     P(娱乐|被预测文档) = P(娱乐|影院,支付宝,云计算) => 调用贝叶斯公式省略分母 --> P(影院,支付宝,云计算|娱乐) * P(娱乐) --> 分子
                                                                      --> P(影院|娱乐)*P(支付宝*娱乐)*P(云计算*娱乐) * P(科技) --> 分子
                                                                      --> 56/121 * 15/121 * 0/121 * 60/80 = 0
                                                                      --> (56+1)/(121+1*4) * (15+1)/(121+1*4) * (
                                                                      0+1)/(121+1*4) * 60/80 = 0.000350208

优点: 有稳定的分类概率
     速度快,效率高,对缺失数据不太敏感,受数据限制,受样本特征独立性假设,特征相似效果不好

     深度学习 CNN 循环神经网络,卷积神经网络 自然语言处理,文本分类,限制(分词效果)

"""
from sklearn.datasets import fetch_20newsgroups, load_boston, load_iris
from sklearn.model_selection import train_test_split  # 分割数据集为训练集和测试集
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # 朴素被颜色


def news_fenlei():
    """" 朴素贝叶斯对20类新闻进行分类 """
    news_data = fetch_20newsgroups(subset="all")  # subset = "all" 获取所有数据
    # 1. 分割数据集为训练集和数据集
    # print(news_data.data)
    # print(news_data.target)
    x_train, x_test, y_train, y_test = train_test_split(news_data.data, news_data.target, test_size=0.3)
    # 2.进行文本特征抽取目的 --> 特征数字化 --> 朴素贝叶斯算法知道每篇文章的特点
    tfidf = TfidfVectorizer(stop_words=[])  # 提高准确率
    # 3.接下来要对训练集和测试集都要进行特征抽取
    # 3.1 训练集特征抽取
    new_x_train = tfidf.fit_transform(x_train)  # 英文默认空格分割
    # print(x_train)
    # print(tfidf.get_feature_names())
    # print("*" * 100)
    # print(new_x_train.toarray())
    # 3.2 对测试集特征抽取
    # 以历史数据为预测依据,不能fit_transform,用原来的标准去分类,相同的矩阵
    new_x_test = tfidf.transform(x_test)  # 不可以fit_transform标准不一不可以,tfidf.get_future_names()
    # 4.朴素贝叶斯算法预测
    mlb = MultinomialNB(alpha=1.0)  # 拉普拉斯平滑系数
    mlb.fit(new_x_train, y_train)  # 开始训练
    print(new_x_test.toarray())
    print("end")
    print("预测测试集当中的文档类别:", mlb.predict(new_x_test)[:50])  # 预测测试集的文档类别,用测试集来分类
    print("测试集当中文档的真实类别", y_test[:50])
    print("朴素贝叶斯算法分类的准确率为", mlb.score(new_x_test, y_test))

    pass


if __name__ == '__main__':
    news_fenlei()
