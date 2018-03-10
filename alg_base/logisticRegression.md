# 逻辑回归（logistic Regression）

## 基本概念
线性模型在解决分类问题的时候健壮性很差，现实中很多时候样本集中包含脏数据，例如电商数据中的刷单行为等。当噪声来的时候，模型很容易走偏。

在分类问题中希望有一个函数能输出分类，可以在线性回归模型外加入一个压缩函数，逻辑回归（logistics model）就是加入了sigmoid函数作为压缩函数的广义线性模型。

## Sigmoid压缩函数
Sigmoid函数由下列公式定义:

![](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D99/sign=a46bd6f1dd33c895a27e9472d01340df/0df3d7ca7bcb0a4659502a5f6f63f6246b60af62.jpg)

Sigmoid函数图例：

![](https://gss3.bdstatic.com/-Po3dSag_xI4khGkpoWK1HF6hhy/baike/w%3D268%3Bg%3D0/sign=ba0ac7a864061d957d46303e43cf6dec/d009b3de9c82d158dfb4e7218a0a19d8bc3e426f.jpg)

Sigmoid函数是对概率p做拟合，是0和1之间的一个值（不包括0和1），当x取0的时候，y取0.5。

其对x的导数可以用自身表示:
![](https://gss1.bdstatic.com/-vo3dSag_xI4khGkpoWK1HF6hhy/baike/s%3D236/sign=375012cedfca7bcb797bc02c88086b3f/64380cd7912397dde41ab3095182b2b7d0a2875f.jpg)

## 决策边界（decision boundary）

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/201704313.png)

## 损失函数
逻辑回归L2损失不是凸函数。

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/201704314.jpg)

逻辑回归损失函数：

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/alg_base/pic/QQ20170817-172826.png)

当y=1时，假定这个样本为正类。如果此时hθ(x)=1,则单对这个样本而言的cost=0,表示这个样本的预测完全准确。那如果所有样本都预测准确，总的cost=0 

但是如果此时预测的概率hθ(x)=0，那么cost→∞。直观解释的话，由于此时样本为一个正样本，但是预测的结果P(y=1|x;θ)=0, 也就是说预测 y=1的概率为0，那么此时就要对损失函数加一个很大的惩罚项。 

当y=0时，推理过程跟上述完全一致，不再累赘。

将以上两个表达式合并为一个，则单个样本的损失函数可以描述为:

![](https://github.com/bobkentt/Learning-machine-from-scratch-pic/blob/master/practice/pic/20170601182340.jpg)

## 损失函数最优化
一般用gradient descent，如果需要分布式可以考虑L-BFGS

## 代码实例
[点击链接进入逻辑回归代码实战](https://github.com/bobkentt/Learning-machine-from-scratch-/blob/master/practice/logistic-regression-practice.md)
