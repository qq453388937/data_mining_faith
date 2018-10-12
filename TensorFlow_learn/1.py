# -*- coding:utf-8 -*-
"""
* 机器学习和深度学习的区别

  * 特征提取方面
    * 机器学习：手动进行特征工程
    * 深度学习：算法自动筛选提取
      * 适合用在难提取特征的图像、语音、自然语言领域
  * 数据量的大小
    * 机器学习：数据量偏小
    * 深度学习：数据量非常大
      * 导致计算时间比较长，需要各种计算设备
  * 算法上的区别
    * 机器学习：朴素贝叶斯、决策树等
    * 深度学习：神经网络
* 应用场景

  * 图像、语音、文本  自然语言
  * 其它都可以去进行尝试

* TensorFlow是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Operation）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。


* Tensorflow程序当中的重要组成部分

  * **一个构建图阶段**：图（程序）的定义
    * 张量：TensorFlow 中的基本数据对象
    * 节点(OP): 指的也是运算操作（也包括也一些不仅是提供）
  * **一个执行图阶段**：会话去运行图程序
"""

import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g = tf.Graph()  # 创建一张图

# 实现加法
t1 = tf.constant(10, name="mmd")  # op名称
t2 = tf.constant(20)

temp = 0
c = temp + t1  # 重载为op加法

# 这个数据,在图当中没有明确定义好数据的内容
# plt = tf.placeholder(dtype=tf.float32, shape=[3, 4])  # 3行4列
plt = tf.placeholder(dtype=tf.float32, shape=[None, 2])  # 不固定行2列
plt1 = tf.placeholder(dtype=tf.float32, shape=[None, 2])  # 不固定行2列

sum_number = tf.add(t1, t2)
print(g)
with g.as_default():
    con_g = tf.constant(666)  # 定义是op去定义的,装有数据的,所以打印出来是Tensor
    print(con_g.graph)

print(tf.get_default_graph())
with tf.Session() as sess:  # graph=""
    # 一般在会话当中序列events文件
    file_writer = tf.summary.FileWriter('./', graph=sess.graph)
    print(t1.graph)
    print(t2.graph)
    print(sum_number.graph)

    # print(sess.run(sum_number))
    # print(sess.run(c))
    print(sess.run([plt, plt1], feed_dict={plt: [[1, 2], [3, 4], [5, 6]], plt1: [[1, 2], [3, 4], [5, 6]]}))
    # print(sess.run([sum_number, t1, t2]))  # 运行多个列表

# with tf.device("GPU:0")
