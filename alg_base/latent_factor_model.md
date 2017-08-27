# 隐语义模型(latent factor model，LFM)

## 基本思想
还有一种方法，可以对书和物品的兴趣进行分类。对于某个用户，首先得到他的兴趣分类，
然后从分类中挑选他可能喜欢的物品。 总结一下，这个基于兴趣分类的方法大概需要解决3个问题。

1. 如何给物品进行分类?
2. 如何确定用户对哪些类的物品感兴趣，以及感兴趣的程度?
3. 对于一个给定的类，选择哪些属于这个类的物品推荐给用户，以及如何确定这些物品在
一个类中的权重?

## 问题1
如果两个物品被很多用户同时喜欢，那么这两个物品就很有可能属于同一个类。

LFM通过如下公式计算用户u对物品i的兴趣:

```
import numpy as np
# puk度量用户u的兴趣和第k个隐类的关系
pu = np.mat(...)
# qik度量第k个隐类和物品i之间的关系
qi = np.mat(...)
# 用户u对物品i的兴趣
rui = pu.T().dot(qi)


```
# 损失函数
![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/practice/pic/20170827-153201.png)
