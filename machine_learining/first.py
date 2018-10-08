# -*- coding:utf-8 -*-


# Scikit-learn  实现数据集的特征工程
# 机器学习的算法和原理
# 应用Scikit-learn==0.18 实现机器学习算法的应用,结合场景解决实际问题
# 人工智能 > 机器学习(典型问题:垃圾邮件分类)  > 深度学习(图像识别)

# 机器学习的定义: 数据, 自动分析获得规律, 对未知数据进行预测
# 意义: 提高生产效率,量化投资,智能客服
# TensorFlow
# 掌握算法的应用场景, 从某个业务领域切入问题

# 特征工程
# : 专业背景知识和技巧处理数据, 使得特征能在机器学习算法上的发挥更好的作用的过程
""" 数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已"""
# 需要筛选处理一些合适的特征

# 数据集的构成  :  特征值(事物的特点) + 目标值(预测的结果)  (重要!!!)

# 特征工程包含3个内容: 1.特征抽取  2.特征预处理  3. 特征降维
# 1.特征抽取: 将任意数据( 如文本或图像,类别 ) 转换为可用于机器学习的数字特征!!!!! (重要!!!)  字典,文本,图像


import sklearn, jieba
#      特征抽取模块
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def countvec():
    """ 文本特征提取 """
    # 实例化count
    count = CountVectorizer()  # 没有sparse = False
    #                              穿进去列表               中文有问题,只能以符号隔开
    data = count.fit_transform(['Life is is short, i like python 大大', 'life is too long, i dislike python'])
    print(count.get_feature_names())  # 文本抽取返回单词列表(重复的词只算一次)  单个字母没影响,默认过滤不计入
    print(data.toarray())


from sklearn.feature_extraction import DictVectorizer


def dictvec():
    """ 对字典数据特征抽取 """
    """ 目的对特征当中有类别的信息做处理,处理: one-hot编码->最好选择"""
    # 实例化
    dict_vec = DictVectorizer(sparse=False)
    # 3个样本的特征数据 (字典形式)
    dict_data = [{'city': '北京', 'temperature': 100},
                 {'city': '上海', 'temperature': 60},
                 {'city': '深圳', 'temperature': 30}]
    # 调用fit_transform

    data = dict_vec.fit_transform(dict_data)  # 默认返回sparse矩阵(目的节省空间->了解)
    print(dict_vec.get_feature_names())  # 'city=上海', 'city=北京', 'city=深圳', 'temperature']
    print(data)  # sparse矩阵
    """ one-hot 编码 """
    # print(data.toarray())   or  sparse=False 即可

    # ['city=上海','city=北京','city=深圳','temperature']


def fenci():
    def cut_word(s1, s2, s3):
        c1 = jieba.cut(s1)
        c2 = jieba.cut(s2)
        c3 = jieba.cut(s3)
        print(c3)  # <generator object Tokenizer.cut at 0x10d1cb8e0>
        # 先将着三个转换成列表,变成以空格隔开的字符串
        ct1 = " ".join(list(c1))
        ct2 = " ".join(list(c2))
        ct3 = " ".join(list(c3))
        print(ct3)  # 如果 只用 一种 方式 了解 某样 事物 ， 你 就 不会 真正 了解 它 。 了解 事物 真正 含义 的 秘密 取决于 如何 将 其 与 我们 所 了解 的 事物 相 联系 。
        return ct1, ct2, ct3

    s1 = "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。"
    s2 = "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。"
    s3 = "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"
    ret1, ret2, ret3 = cut_word(s1, s2, s3)
    # 实例化对象
    count = CountVectorizer(stop_words=["不会", "不要", "绝对"])
    # 对分词特征进行抽取
    data = count.fit_transform([ret1, ret2, ret3])
    print(count.get_feature_names())
    print(data.toarray())
    """ stop_words 停止词,这些词不能反映文章主题,词语性质比较中性,因为,所以 """


from sklearn.feature_extraction.text import TfidfVectorizer


def TfIdfvector():
    """
        tdidf 作用: 用以评估一字词对于一个文件集或者一个语料库中的其中一份文件的重要程度
        为了处理这种同一个词在很多文章出现次数较高
        tf 词频 是一个词语在文章中出现的频率 出现次数除以总次数 5/100=0.05
        逆文档频率 idf = lg(10000000/10000)=3  lg(一千万/一万) = 3
        tfidf = 3 * 0.05 => 0.15
    """

    def cut_word(s1, s2, s3):
        c1 = jieba.cut(s1)
        c2 = jieba.cut(s2)
        c3 = jieba.cut(s3)
        print(c1)  # <generator object Tokenizer.cut at 0x10e934bf8> 生成器
        # 先将着三个转换成列表,变成以空格隔开的字符串
        ct1 = " ".join(list(c1))
        ct2 = " ".join(list(c2))
        ct3 = " ".join(list(c3))
        print(ct1)  # 如果 只用 一种 方式 了解 某样 事物 ， 你 就 不会 真正 了解 它 。 了解 事物 真正 含义 的 秘密 取决于 如何 将 其 与 我们 所 了解 的 事物 相 联系 。
        return ct1, ct2, ct3

    s1 = "今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。"
    s2 = "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。"
    s3 = "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"
    ret1, ret2, ret3 = cut_word(s1, s2, s3)
    print(ret1, ret2, ret3)
    # 实例化对象
    tfidf = TfidfVectorizer(stop_words=["不会", "不要", "绝对"])  # 对每篇文章的重要性排序,找到前N个重要词
    """ 分类机器算法前期处理方式 """
    # 对分词特征进行抽取
    data = tfidf.fit_transform([ret1, ret2, ret3])
    print(tfidf.get_feature_names())
    print(data.toarray())


if __name__ == '__main__':
    # dictvec()
    # countvec()
    # fenci()
    TfIdfvector()
