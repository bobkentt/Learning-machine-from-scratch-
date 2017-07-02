# K-means++

## 核心思想

K-means++其实核心思想也是选择更远的点作为初始化中心点，不同的是加入了随机化的处理。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering2/1.jpeg)

虽然单个噪声点的概率大，但我们的目的是使初始化点不落入噪声点，

如下图所示，除了最右侧的一个噪声点，落入其它点都可以，

而经过了归一化的处理之后，落入其它正常点的概率是远远大于落入噪声点的概率，这就是K-means算法的核心思想。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering/14.png)

## 效果

K-means++ always attain an O(logk) approximation to optimal k-means solution in expectation.

随机化算法是一个非常常用的技巧，

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering2/2.jpeg)

## K-means++ running time

K-means++ initialization: O(nd) and an pass over data to
select next center. so O(nkd) time in total.

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering2/3.jpeg)

**上面的图对面试十分重要**

## 怎么选择k到底是多少？

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/clustering2/4.jpeg)

* k值越大，目标函数误差期望应该越小，尝试增大k的值，当k的值增大，到一定程度时下降越来越慢，找到拐点。

* 交叉验证数据集，在训练集中训练不同的k，出来不同模型。然后在测试集上验证，找出最好k所在模型。

```
对于大部分的机器学习，怎么选择合适合适的参数?（面试）

1.聚类有多少个类别？

2.神经网络中有多少个神经元？

标准答案用交叉验证，cross-validation做

```
